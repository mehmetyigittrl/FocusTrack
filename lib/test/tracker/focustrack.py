import math
from collections import deque

from lib.models.focustrack import build_focustrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.test.tracker.gpu_crop import gpu_sample_target
from lib.utils.box_ops import clip_box
from lib.utils.misc import NestedTensor
from lib.utils.tir_stack import stack_pair, stack_pair_gpu
from lib.utils.ce_utils import generate_mask_cond
from lib.train.data.grid_generator import unwarp_bboxes, unwarp_bboxes_batch, QPGrid
from lib.utils.box_ops import box_cxcywh_to_xywh
from lib.test.tracker.vis_utils import vis_attn_maps, vis_feature_maps, vis_attn_distance


class FocusTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(FocusTrack, self).__init__(params)
        network = build_focustrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor(self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), 
                                     centered=True).cuda()
        
        self.use_grid = self.cfg.TEST.SEARCH.USE_GRID
        if self.use_grid:
            loss_dict = dict()
            for loss_name, loss_weight in zip(self.cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.NAMES, self.cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.WEIGHTS):
                loss_dict[loss_name] = loss_weight
            self.search_grid_generator = QPGrid(
                amplitude_scale=1,
                bandwidth_scale=self.cfg.TEST.SEARCH.GRID.GENERATOR.BANDWIDTH_SCALE,
                grid_type='qp',
                zoom_factor=self.cfg.TEST.SEARCH.GRID.GENERATOR.ZOOM_FACTOR,
                loss_dict=loss_dict,
                grid_shape=self.params.cfg.TEST.SEARCH.GRID.SHAPE
            )
            self.out_shape = (self.params.cfg.TEST.SEARCH.SIZE, self.params.cfg.TEST.SEARCH.SIZE)

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.base_factor = self.params.search_factor
        self.search_factor = self.params.search_factor
        self.flag = 1
        self.use_hann=True

        # gpu crop / tir stack / dynamic template gates
        self.use_gpu_crop = self.cfg.TEST.GPU_CROP
        self.search_min_crop_size = self.cfg.TEST.SEARCH.MIN_CROP_SIZE
        self.template_min_crop_size = self.cfg.TEST.TEMPLATE.MIN_CROP_SIZE
        self.use_tir_stack = self.cfg.DATA.TIR_STACK.USE
        self.tir_delta = self.cfg.DATA.TIR_STACK.DELTA
        self.tir_template_stack = self.cfg.DATA.TIR_STACK.TEMPLATE_STACK
        self.use_dynamic_template = self.cfg.TEST.DYNAMIC_TEMPLATE
        self.template_update_score = self.cfg.TEST.TEMPLATE_UPDATE_SCORE
        self.template_update_prob = self.cfg.TEST.TEMPLATE_UPDATE_PROB
        self.template_update_milestones = self.cfg.TEST.TEMPLATE_UPDATE_MILESTONES
        self.mean = self.cfg.DATA.MEAN
        self.std = self.cfg.DATA.STD
        self.tir_hist_gpu = deque(maxlen=max(1, self.tir_delta))
        self.tir_hist_np = deque(maxlen=max(1, self.tir_delta))

    def initialize(self, image, info: dict):
        # forward the template once
        z_img = image
        if self.use_tir_stack and self.tir_template_stack:
            z_img = stack_pair(image, image)  # no previous frame at init: diff channel is zero
        z_patch_arr, resize_factor, z_amask_arr = sample_target(z_img, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size,
                                                    min_crop_size=self.template_min_crop_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        # dynamic template update state: scale ladder measured against the init box area
        self.last_template_area = float(info['init_bbox'][2]) * float(info['init_bbox'][3])
        self.template_milestone_idx = 0
        # re-init support (e.g. video demo 'r' reset): restore the base search state,
        # not whatever the escalation state machine left behind mid-video
        self.search_factor = self.base_factor
        self.use_hann = True
        self.flag = 1
        self.tir_hist_np = deque([image], maxlen=max(1, self.tir_delta))
        if self.use_gpu_crop and not self.use_grid and not self.debug and self.use_tir_stack:
            self.tir_hist_gpu = deque([torch.from_numpy(image).cuda().float().permute(2, 0, 1)],
                                      maxlen=max(1, self.tir_delta))
        else:
            self.tir_hist_gpu = deque(maxlen=max(1, self.tir_delta))

        self.box_mask_z = None

        # save states
        self.init_box = info['init_bbox']
        self.state = info['init_bbox']
        self.former_box = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
                
        # 计算 max_search_region
        self.W, self.H = image.shape[1], image.shape[0]
        
        self.max_search_area_factor = self.cfg.TEST.MAX_SEARCH_FACTOR
        self.step = self.cfg.TEST.ENLARGE_STEP
        self.T_logits = self.cfg.TEST.T_LOGITS
        self.T_score = self.cfg.TEST.T_SCORE
        self.T_cls_prob = self.cfg.TEST.T_CLS_PROB

    def _esc_min_crop(self):
        """ Factor-relative min crop floor: scale the tiny-target clamp with the search
        factor so SRA escalation actually enlarges the search region. A fixed 64 px floor
        is hit at both factor 6 and 8 for 3x5 px boxes, making escalation a no-op. """
        return math.ceil(self.search_min_crop_size * self.search_factor / self.base_factor)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        W_H = np.array([W, H])
        self.frame_id += 1
        self.former_box = self.state
        # TIR prev frame read BEFORE the history append below, so the search stack and any
        # dynamic template refresh both pair (I_t, I_{t-delta}) like the training sampler
        prev_np = self.tir_hist_np[0] if self.use_tir_stack and self.tir_hist_np else image
        
        if self.use_gpu_crop and not self.use_grid and not self.debug:
            # fast path: upload once, crop + normalize on GPU (no CPU sample_target/preprocessor)
            frame_cuda = torch.from_numpy(image).cuda().float().permute(2, 0, 1)  # (C,H,W) 0..255
            if self.use_tir_stack:
                prev_frame = self.tir_hist_gpu[0] if self.tir_hist_gpu else frame_cuda
                search_frame = stack_pair_gpu(frame_cuda, prev_frame)
                self.tir_hist_gpu.append(frame_cuda)
            else:
                search_frame = frame_cuda
            x_tensor, resize_factor, x_amask = gpu_sample_target(search_frame, self.state, self.search_factor,
                                                                 self.params.search_size,
                                                                 self._esc_min_crop(), self.mean, self.std)
            x_dict = NestedTensor(x_tensor, x_amask.unsqueeze(0))
        elif not self.use_grid:
            # CPU fallback: stack the TIR frame pair first so the search region has the same
            # [I_t, I_prev, |diff|] semantics as the GPU fast path and the template branch.
            search_img = image
            if self.use_tir_stack:
                search_img = stack_pair(image, prev_np)
            x_patch_arr, resize_factor, x_amask_arr = sample_target(search_img, self.state, self.search_factor,
                                                                    output_sz=self.params.search_size,
                                                                    min_crop_size=self._esc_min_crop())  # (x1, y1, w, h)
            x_dict = self.preprocessor.process(x_patch_arr, x_amask_arr)   # 经过reshape的search
        else:
            x_patch_arr, x_amask_arr, resize_factor = sample_target(image, self.state, self.search_factor)  # (x1, y1, w, h)
            
            img_shape_ori = x_patch_arr.shape
            img_metas = [
                {'pad_shape': img_shape_ori}
            ]
            img_tensor = torch.tensor(x_patch_arr, dtype=torch.float32).permute(2,0,1)[None,...]
            x_patch_h, x_patch_w, _ = img_shape_ori
            saliency_box = torch.tensor(self.state, dtype=torch.float32)
            saliency_box[0] = x_patch_w / 2 - saliency_box[2] / 2
            saliency_box[1] = x_patch_h / 2 - saliency_box[3] / 2

            grid, saliency = self.search_grid_generator.forward(img_tensor.cuda(), img_metas, saliency_box[None,None,:].cuda(), self.out_shape, jitter=0, mode='center')
            x_patch_arr = F.grid_sample(img_tensor.cuda(), grid, align_corners=True)
            x_patch_arr = x_patch_arr.cpu()[0].permute(1,2,0).numpy()    # (256, 256, 3)
        
        
        if self.use_grid:
            x_dict = self.preprocessor.process(x_patch_arr, x_amask_arr)   # 经过reshape的search

        if self.use_tir_stack:
            self.tir_hist_np.append(image)

        with torch.no_grad():
            out_dict = self.network.forward(template=self.z_dict1.tensors, search=x_dict.tensors, training=False)
        
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = out_dict['score_map']
        if self.use_hann:
            response = response * self.output_window

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        if not self.use_grid:
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        else:
            pred_box = pred_boxes.mean(dim=0, keepdim=True)  # (cxywh)
            search_whwh = torch.tensor([self.params.search_size, self.params.search_size, self.params.search_size, self.params.search_size], device=pred_box.device)

            pred_box = torch.cat([pred_box[:,:2]-pred_box[:,2:]/2, pred_box[:,:2]+pred_box[:,2:]/2], dim=-1)   # (xywh)
            unwraped_box = unwarp_bboxes(pred_box * search_whwh, grid[0], img_shape_ori)
            unwraped_box = torch.cat([(unwraped_box[:,2:]+unwraped_box[:,:2])/2, unwraped_box[:,2:]-unwraped_box[:,:2]], dim=-1).cpu()
            if self.debug:
                # only display top-k box
                feat_x = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
                feat_y = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
                feat_xyxy = torch.tensor([feat_x, feat_y, feat_x, feat_y])
                topk_score, topk_index = torch.topk(pred_score_map.squeeze().flatten(), 4)
                topk_index2d = torch.stack([topk_index // pred_score_map.shape[-1], topk_index % pred_score_map.shape[-1]], dim=1)
                topk_pred_boxes_raw = self.cal_box_from_coord(out_dict, topk_index2d.to(out_dict['score_map'].device), resize_factor, H , W, reduce='raw')
                topk_pred_boxes_raw_xyxy = torch.cat([topk_pred_boxes_raw[:,:2]-topk_pred_boxes_raw[:,2:]/2, topk_pred_boxes_raw[:,:2]+topk_pred_boxes_raw[:,2:]/2], dim=-1)
                unwraped_topk_pred_boxes_raw = unwarp_bboxes(topk_pred_boxes_raw_xyxy * search_whwh.cuda(), grid[0], img_shape_ori)
                full_grid = F.interpolate(grid.permute(0,3,1,2), scale_factor=resize_factor)

                # wrong gt_box to make program work
                gt_box = transform_image_to_crop(torch.tensor(info['gt_bbox']), torch.tensor(self.state), resize_factor, x_patch_h)
                pred_boxes_ori = unwraped_topk_pred_boxes_raw.cpu()
                pred_boxes_ori = torch.cat([(pred_boxes_ori[:,2:]+pred_boxes_ori[:,:2])/2, pred_boxes_ori[:,2:]-pred_boxes_ori[:,:2]], dim=-1)
                pred_boxes_ori = torch.stack([torch.tensor(clip_box(self.map_box_back_fullsize(pred_boxes_ori_single.tolist(), x_patch_h), H, W, margin=10))for pred_boxes_ori_single in pred_boxes_ori], dim=0)
            self.state = clip_box(self.map_box_back_fullsize(unwraped_box[0].tolist(), x_patch_h), H, W, margin=10)
        
      
        if self.cfg.TEST.USE_REGION_ADJUST or self.use_dynamic_template:
            # presence prob: sigmoid over the 2-way cls logits (last class = target present);
            # peak raw score (before hann); one stacked tensor -> a single host sync per frame
            cls_prob = torch.sigmoid(out_dict['logits'])[0, -1]
            max_score = out_dict['score_map'].max()
            cls_prob, max_score = torch.stack([cls_prob, max_score]).cpu().tolist()

        if self.cfg.TEST.USE_REGION_ADJUST:
            # OR-gating: escalate on either weak presence or a flat score map, catching the
            # single-signal failures the old AND-gated softmax logic missed
            self.flag = cls_prob
            if cls_prob < self.T_cls_prob or max_score < self.T_score:
                self.search_factor = min(self.search_factor + self.step, self.max_search_area_factor)
            else:
                self.search_factor = self.base_factor
            self.use_hann = self.search_factor <= self.base_factor

        if self.use_dynamic_template and max_score > self.template_update_score \
                and cls_prob > self.template_update_prob:
            # safe scale-aware template refresh in place (single template z_dict1);
            # never on low-confidence frames - that is the whole point
            area = float(self.state[2]) * float(self.state[3])
            milestones = self.template_update_milestones
            milestone = milestones[min(self.template_milestone_idx, len(milestones) - 1)]
            if area >= milestone * self.last_template_area:
                # milestone ladder 2x/4x/8x/16x; the idx is capped at the last rung so
                # growth beyond 16x keeps refreshing at every further 16x
                z_img = image
                if self.use_tir_stack and self.tir_template_stack:
                    z_img = stack_pair(image, prev_np)
                z_patch_arr, _, z_amask_arr = sample_target(z_img, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size,
                                                            min_crop_size=self.template_min_crop_size)
                with torch.no_grad():
                    self.z_dict1 = self.preprocessor.process(z_patch_arr, z_amask_arr)
                self.z_patch_arr = z_patch_arr
                self.last_template_area = area
                self.template_milestone_idx = min(self.template_milestone_idx + 1, len(milestones) - 1)
                
        
        if self.debug:
            if not self.use_visdom:
                # green for gt
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                x2, y2, w2, h2 = info['gt_bbox'].tolist()
                cv2.rectangle(image_BGR, (int(x2),int(y2)), (int(x2+w2),int(y2+h2)), color=(0, 255, 0), thickness=2)
                
                # red for base
                x3, y3, w3, h3 = PRED_BB[self.frame_id]
                cv2.rectangle(image_BGR, (int(x3),int(y3)), (int(x3+w3),int(y3+h3)), color=(0,0,255), thickness=2)
                
                # blue for pred
                x1, y1, w1, h1 = self.state
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w1),int(y1+h1)), color=(255,0,0), thickness=2)
                
                cv2.putText(image_BGR, 'GroundTruth', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image_BGR, 'FocusTrack', (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(image_BGR, 'Base', (510, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
                
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state, PRED_BB[self.frame_id], 
                                       self.frame_id, self.search_factor, self.flag), 'Tracking', 1, 'Tracking')
                
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                self.visdom.register(response.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'response')
                # self.visdom.register((image, self.state, *pred_boxes_ori[~((pred_boxes_ori == torch.tensor(self.state)).all(1))]), 'Tracking', 1, 'Tracking')

                
                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                    
                if self.use_grid:
                    x_patch_arr_ori = img_tensor[0].permute(1,2,0).numpy()
                    x_patch_arr_resize = cv2.resize(x_patch_arr_ori, (self.params.search_size, self.params.search_size))
                    self.visdom.register(torch.from_numpy(x_patch_arr_resize).permute(2, 0, 1), 'image', 1, 'resized_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False 
                        break
                
                if 'full_mask' in out_dict:
                    search_size = self.cfg.DATA.SEARCH.SIZE * self.cfg.DATA.SEARCH.SIZE
                    search_mask = out_dict['full_mask'][:,-search_size: ]
                    if search_mask.dim() == 2:
                        n, hw = search_mask.size()
                        search_mask = search_mask.reshape(n, self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE)
    
                    mask = search_mask.permute(1, 2, 0).cpu().numpy()    
                    
                    
                    heatmap = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)
                    x_patch = x_patch_arr
                    alpha=0.3
                    overlay = cv2.addWeighted(x_patch, 1 - alpha, heatmap, alpha, 0)
                    show_box = box_cxcywh_to_xywh(torch.tensor(pred_boxes.mean(dim=0))) * self.cfg.TEST.SEARCH.SIZE       
                    cv2.rectangle(overlay,
                                    (int(show_box[0]), int(show_box[1])),
                                    (int(show_box[0] + show_box[2]), int(show_box[1] + show_box[3])), (255, 0, 0), 2)
                    self.visdom.register(torch.from_numpy(overlay).permute(2, 0, 1), 'image', 1, 'full_mask')
                    
                    
                    self.visdom.register(out_dict['feat_mask'].view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'feat_mask')
        

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
    
    def compute_iou(self, box1, box2):
        """ Compute IoU between two boxes in (x, y, width, height) format.
        args:
            box1: list or tensor of format [x, y, width, height]
            box2: list or tensor of format [x, y, width, height]
        returns:
            iou: float - intersection over union value
        """
        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)

        # Convert (x, y, width, height) to (x1, y1, x2, y2)
        box1_x1y1 = box1[0:2]
        box1_x2y2 = box1[0:2] + box1[2:4]
        box2_x1y1 = box2[0:2]
        box2_x2y2 = box2[0:2] + box2[2:4]

        # Calculate intersection
        inter_x1 = torch.max(box1_x1y1[0], box2_x1y1[0])
        inter_y1 = torch.max(box1_x1y1[1], box2_x1y1[1])
        inter_x2 = torch.min(box1_x2y2[0], box2_x2y2[0])
        inter_y2 = torch.min(box1_x2y2[1], box2_x2y2[1])
        inter_area = torch.max(inter_x2 - inter_x1, torch.tensor(0.0)) * torch.max(inter_y2 - inter_y1, torch.tensor(0.0))

        # Calculate union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area

        # Compute IoU
        iou = inter_area / union_area
        return iou.item()

    

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
    def map_box_back_fullsize(self, pred_box: list, full_size: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * full_size
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output)
            # 可以多注册几个
            )

        self.enc_attn_weights = enc_attn_weights

    def cal_box_from_coord(self, out_dict, coord,  resize_factor, H , W, reduce='mean'):
        """
        coord - (2) or (num_box, 2) dtype:long
        """
        pred_boxes= self.network.box_head.cal_bbox_from_idx(coord, out_dict['size_map'], out_dict['offset_map'])
        
        pred_boxes = pred_boxes.view(-1, 4)
        if reduce == 'mean':
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            return clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        elif reduce == 'none':
            boxes = []            
            pred_boxes = (pred_boxes * self.params.search_size / resize_factor).tolist()
            for box in pred_boxes:
                boxes.append(torch.tensor(clip_box(self.map_box_back(box, resize_factor), H, W, margin=10)))
            return torch.stack(boxes)
        elif reduce == 'raw':
            return pred_boxes
        else:
            raise NotImplementedError

def get_tracker_class():
    return FocusTrack
