# This is a minimally modified version of the eval function from the SeqNet repository (https://github.com/serend1p1ty/SeqNet/blob/master/eval_func.py)
# Changes:
# - Removed code related to CBGM (Context Bipartite Graph Matching)
# - Adjusted top-k accuracy calculation to only consider top-1 accuracy
# - Clarified function docstring and added recall rate scaling explanation

import os.path as osp

import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from testing.km import run_kuhn_munkres

# from utils.utils import write_json

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def eval_search_prw(
    gallery_dataset,
    query_dataset,
    gallery_dets,
    gallery_feats,
    query_box_feats,
    query_dets,
    query_feats,
    det_thresh=0.5,
    ignore_cam_id=True,
    cbgm=False,
 ):
    """
    Evaluate person search performance on PRW dataset.

    Args:
        gallery_dataset (Dataset): dataset containing gallery images.
        query_dataset (Dataset): dataset containing query images.
        gallery_dets (list of ndarray): n_det x [x1, x2, y1, y2, score] per image.
        gallery_feats (list of ndarray): n_det x D features per image.
        query_box_feats (list of ndarray): D dimensional features per query image.
        det_thresh (float): filter out gallery detections whose scores below this.
        ignore_cam_id (bool): whether to ignore camera ID during evaluation. If set to False,
                            gallery images from the same camera as the query will be excluded. Default: True.
    """
    assert len(gallery_dataset) == len(gallery_dets)
    assert len(gallery_dataset) == len(gallery_feats)
    assert len(query_dataset) == len(query_box_feats)

    annos = gallery_dataset.annotations
    # fast lookup map for anno by image name (used to recover pids for visualizations)
    name_to_anno = {a['img_name']: a for a in annos}
    name_to_det_feat = {}
    for anno, det, feat in zip(annos, gallery_dets, gallery_feats):
        name = anno["img_name"]
        scores = det[:, 4].ravel()
        inds = np.where(scores >= det_thresh)[0]
        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds], feat[inds])

    aps = []
    accs = []
    topk = [1] # we are only interested in top-1 accuracy
    ret = {"image_root": gallery_dataset.img_prefix, "results": []}

    print("[Eval 3/3] Search ranking over query set...")
    query_iter = tqdm(
        range(len(query_dataset)),
        total=len(query_dataset),
        desc="Eval 3/3 | ranking",
        leave=True,
        dynamic_ncols=True,
    )

    for i in query_iter:
        y_true, y_score = [], []
        imgs, rois = [], []
        count_gt, count_tp = 0, 0

        feat_p = query_box_feats[i].ravel()

        query_imname = query_dataset.annotations[i]["img_name"]
        query_roi = query_dataset.annotations[i]["boxes"]
        query_pid = query_dataset.annotations[i]["pids"]
        query_cam = query_dataset.annotations[i]["cam_id"]

        # Find all occurence of this query
        gallery_imgs = []
        for x in annos:
            if query_pid in x["pids"] and x["img_name"] != query_imname:
                gallery_imgs.append(x)
        query_gts = {}
        for item in gallery_imgs:
            query_gts[item["img_name"]] = item["boxes"][item["pids"] == query_pid]

        # Construct gallery set for this query
        if ignore_cam_id:
            gallery_imgs = []
            for x in annos:
                if x["img_name"] != query_imname:
                    gallery_imgs.append(x)
        else:
            gallery_imgs = []
            for x in annos:
                if x["img_name"] != query_imname and x["cam_id"] != query_cam:
                    gallery_imgs.append(x)

        name2sim = {}
        sims = []
        
        # added to integrate Contextual Bipartite Graph Matching (CBGM) 
        imgs_cbgm = []
        
        # 1. Go through all gallery samples
        for item in gallery_imgs:
            gallery_imname = item["img_name"]
            # some contain the query (gt not empty), some not
            count_gt += gallery_imname in query_gts
            # compute distance between query and gallery dets
            if gallery_imname not in name_to_det_feat:
                continue
            det, feat_g = name_to_det_feat[gallery_imname]
            # get L2-normalized feature matrix NxD
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2])
            # compute cosine similarities
            sim = feat_g.dot(feat_p).ravel()

            if gallery_imname in name2sim:
                continue
            name2sim[gallery_imname] = sim
            sims.extend(list(sim))
            imgs_cbgm.extend([gallery_imname] * len(sim))
            
        if cbgm:
            # -------- Context Bipartite Graph Matching (CBGM) ------- #
            k1, k2 = 30, 4 # values used in the original paper
            
            sims = np.array(sims)
            imgs_cbgm = np.array(imgs_cbgm)
            # only process the top-k1 gallery images for efficiency
            inds = np.argsort(sims)[-k1:]
            imgs_cbgm = set(imgs_cbgm[inds])
            for img in imgs_cbgm:
                sim = name2sim[img]
                det, feat_g = name_to_det_feat[img]
                # only regard the people with top-k2 detection confidence
                # in the query image as context information
                qboxes = query_dets[i][:k2]
                qfeats = query_feats[i][:k2]
                assert (
                    query_roi - qboxes[0][:4]
                ).sum() <= 0.001, "query_roi must be the first one in pboxes"

                # build the bipartite graph and run Kuhn-Munkres (K-M) algorithm
                # to find the best match
                graph = []
                for indx_i, pfeat in enumerate(qfeats):
                    for indx_j, gfeat in enumerate(feat_g):
                        graph.append((indx_i, indx_j, (pfeat * gfeat).sum()))
                km_res, max_val = run_kuhn_munkres(graph)

                # revise the similarity between query person and its matching
                for indx_i, indx_j, _ in km_res:
                    # 0 denotes the query roi
                    if indx_i == 0:
                        sim[indx_j] = max_val
                        break

        for gallery_imname, sim in name2sim.items():
            det, feat_g = name_to_det_feat[gallery_imname]
            # assign label for each det
            label = np.zeros(len(sim), dtype=np.int32)
            if gallery_imname in query_gts:
                gt = query_gts[gallery_imname].ravel()
                w, h = gt[2] - gt[0], gt[3] - gt[1]
                iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(sim)[::-1]
                sim = sim[inds]
                det = det[inds]
                # only set the first matched det as true positive
                for j, roi in enumerate(det[:, :4]):
                    if _compute_iou(roi, gt) >= iou_thresh:
                        label[j] = 1
                        count_tp += 1
                        break
            y_true.extend(list(label))
            y_score.extend(list(sim))
            imgs.extend([gallery_imname] * len(sim))
            rois.extend(list(det))

        # 2. Compute AP for this query (need to scale by recall rate)
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        assert count_tp <= count_gt

        # NOTE: I've added a guard to check if count_gt == 0, since then a division by count_gt is perfomed
        if count_gt == 0:
            recall_rate = 0.0
            ap = 0.0
        else:
            # Important: at the pedestrian detection stage, the model might have missed the person (failed to detect a box with IoU > 0.5).
            # To penalize the model for these False Negatives at the detection stage, scale the AP by recall (the ratio of found matches to total ground truth matches).
            # E.g. if the detector missed the person entirely 50% of the time, the final AP score is cut in half.
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
        aps.append(ap)
        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        accs.append([min(1, sum(y_true[:k])) for k in topk])
        # 4. Save result for JSON dump
        new_entry = {
            "query_img": str(query_imname),
            "query_roi": list(map(float, list(query_roi.squeeze()))),
            # store the query pid (may be array-like in annotations)
            "query_pid": int(np.asarray(query_pid).reshape(-1)[0]),
            "query_gt": query_gts,
            "gallery": [],
        }
        # only save top-10 predictions
        for k in range(10):
            img_name_k = str(imgs[inds[k]])
            roi_k = list(map(float, list(rois[inds[k]])))
            # try to assign a pid for this detection by IoU with GT boxes in the
            # corresponding annotation. If no matching GT, fall back to unlabeled pid 5555.
            gallery_pid = 5555
            if img_name_k in name_to_anno:
                anno = name_to_anno[img_name_k]
                boxes = np.asarray(anno.get("boxes", []))
                pids = np.asarray(anno.get("pids", []))
                if boxes.size and boxes.ndim >= 2:
                    # find GT box with max IoU
                    best_iou = 0.0
                    best_pid = 5555
                    for j, gt in enumerate(boxes.reshape(-1, 4)):
                        iou_val = _compute_iou(roi_k, gt)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            try:
                                best_pid = int(np.asarray(pids).reshape(-1)[j])
                            except Exception:
                                best_pid = 5555
                    if best_iou >= 0.5:
                        gallery_pid = best_pid

            new_entry["gallery"].append(
                {
                    "img": img_name_k,
                    "roi": roi_k,
                    "score": float(y_score[k]),
                    "correct": int(y_true[k]),
                    "pid": int(gallery_pid),
                }
            )
        ret["results"].append(new_entry)

    print("search ranking:")
    mAP = np.mean(aps)
    print("  mAP = {:.2%}".format(mAP))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print("  top-{:2d} = {:.2%}".format(k, accs[i]))

    # write_json(ret, "vis/results.json")

    ret["mAP"] = np.mean(aps)
    ret["accs"] = accs
    return ret