
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from PIL import Image

def run_srl_loop(scene, gaussians, pipe, background, iteration, logger):
    """
    Executes the Semantic Refinement Loop (SRL).
    1. Renders semantic maps for all training views.
    2. Filters based on confidence (max probability >= 0.95).
    3. Updates the ground truth object masks in the dataset.
    """
    logger.info(f"\n[ITER {iteration}] Starting Semantic Refinement Loop (SRL)...")
    
    # Ensure model is in eval mode for rendering
    gaussians.eval()
    
    train_cameras = scene.getTrainCameras()
    modules = __import__('gaussian_renderer')
    
    total_pixels = 0
    updated_pixels = 0
    
    with torch.no_grad():
        for idx, view in enumerate(tqdm(train_cameras, desc="SRL: Refining Masks")):
            # Render the view
            render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
            
            # Get rendered semantics: [1, H, W, NumClasses] -> [NumClasses, H, W]
            semantics = render_pkg["render_semantics"]
            # semantics is usually [1, H, W, C], we need to permute to [1, C, H, W] for softmax if it's logits
            # But wait, render_semantics might already be probabilities or logits?
            # In train.py: object_loss = CrossEntropy(semantics.permute(0,3,1,2), ...)
            # So semantics is [1, H, W, C] logits.
            
            logits = semantics.permute(0, 3, 1, 2) # [1, C, H, W]
            probs = F.softmax(logits, dim=1) # [1, C, H, W]
            
            # Get max probability and corresponding label
            max_probs, predicted_labels = torch.max(probs, dim=1) # [1, H, W]
            
            # Squeeze batch dim
            max_probs = max_probs.squeeze(0) # [H, W]
            predicted_labels = predicted_labels.squeeze(0) # [H, W]
            
            # Confidence threshold
            CONFIDENCE_THRESHOLD = 0.95
            high_confidence_mask = max_probs >= CONFIDENCE_THRESHOLD
            
            # Get original mask (on CPU usually, but let's check where it is)
            # view.object_mask is a torch tensor.
            # In train.py it is moved to cuda: view.object_mask.cuda()
            # Here we want to update the source of truth which is view.object_mask
            
            # Ensure everything is on the same device
            device = view.object_mask.device
            if view.object_mask.device != predicted_labels.device:
                # If original mask is on CPU, move prediction to CPU
                # Usually view.object_mask is on CPU (from Camera init)
                predicted_labels = predicted_labels.to(device)
                high_confidence_mask = high_confidence_mask.to(device)
            
            # Update the mask
            # We only update pixels where confidence is high.
            # Note: We should be careful not to overwrite valid labels with "background" (0) if that's not desired,
            # but if the model is confident it's background, maybe we should?
            # The proposal says: "replace the initial 2D object masks... with newly generated 2D masks"
            # "Only those pixels whose ID scores highly confident... will remain, while all other pixels... will be flagged for refinement"
            # This implies we might want to keep the OLD label if the NEW one is not confident?
            # Or maybe we mark low confidence pixels as "ignore" (0 or 255)?
            # "The newly generated 2D masks with high confidence will then replace the initial 2D object masks"
            # This suggests:
            # NewMask = PredictedLabel if Confidence > 0.95 else OldLabel (or Ignore?)
            # Let's assume we replace with prediction if confident, otherwise keep old.
            
            # However, the proposal also says "flagged for refinement".
            # For simplicity in this first pass: Update ONLY if confident.
            
            original_mask = view.object_mask
            
            # Check shapes
            if original_mask.shape != predicted_labels.shape:
                # Resize predicted to match original if needed (e.g. if render resolution differs)
                # But usually render resolution matches view resolution scale.
                # Let's skip resizing for now and assume match.
                pass

            # Update
            # view.object_mask[high_confidence_mask] = predicted_labels[high_confidence_mask]
            # We need to cast predicted_labels to the same type as object_mask (usually uint8 or long)
            predicted_labels = predicted_labels.type(original_mask.dtype)
            
            # Count stats
            # updated_count = (original_mask[high_confidence_mask] != predicted_labels[high_confidence_mask]).sum().item()
            # updated_pixels += updated_count
            # total_pixels += original_mask.numel()
            
            # Perform update
            view.object_mask = torch.where(high_confidence_mask, predicted_labels, original_mask)
            
            # Optional: Save debug images for the first few views
            if idx < 5:
                debug_dir = os.path.join(scene.model_path, f"srl_debug_{iteration}")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save original
                Image.fromarray(original_mask.cpu().numpy().astype(np.uint8)).save(os.path.join(debug_dir, f"{view.image_name}_orig.png"))
                # Save new
                Image.fromarray(view.object_mask.cpu().numpy().astype(np.uint8)).save(os.path.join(debug_dir, f"{view.image_name}_new.png"))
                # Save confidence
                Image.fromarray((max_probs.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(debug_dir, f"{view.image_name}_conf.png"))

    logger.info(f"[SRL] Completed. Updated masks for {len(train_cameras)} views.")
    # logger.info(f"[SRL] Changed {updated_pixels} / {total_pixels} pixels ({updated_pixels/total_pixels*100:.2f}%).")
    
    # Switch back to train mode
    gaussians.train()
