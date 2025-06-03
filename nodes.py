import os
import cv2
import sys
import torch
import contextlib
import numpy as np
import comfy.utils
import folder_paths

from collections import deque
from numpy.linalg import norm
from skimage import transform as trans
from insightface.app import FaceAnalysis


MODELS_PATH = folder_paths.models_dir
INSIGHTFACE_DIR = os.path.join(MODELS_PATH, "insightface")
CURRENT_DIR = os.path.dirname(__file__)


class KPS_sma:
    def __init__(self, buffer=10):
        self.buffer = deque(maxlen=buffer)

    def update(self, new_kps):
        self.buffer.append(new_kps)
        return np.mean(self.buffer, axis=0).astype(np.float32)


class KPS_ema:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.prev_kps = None

    def update(self, new_kps):
        self.prev_kps = self.prev_kps * self.alpha + new_kps * (1 - self.alpha) if self.prev_kps is not None else new_kps
        return self.prev_kps


@contextlib.contextmanager
def suppress_output():
    original_stdout = sys.stdout
    sys.stdout = open('nul', 'w')
    yield
    sys.stdout = original_stdout


def cosine_similarity(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim


def estimate_norm(landmarks, image_size=512, zoom=0.8):
    arcface_dst = np.array([
        [38.2946, 51.6963], 
        [73.5318, 51.5014], 
        [56.0252, 71.7366],
        [41.5493, 92.3655], 
        [70.7299, 92.2041]
        ], dtype=np.float32)
    
    assert landmarks.shape == (5, 2)

    ratio = float(image_size) / 128.0
    diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x

    dst_center = np.mean(dst, axis=0)
    dst = (dst - dst_center) * zoom + dst_center

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    return tform.params[0:2, :]


def norm_crop(image, landmarks, image_size=512, zoom=1.0):
    M = estimate_norm(landmarks, image_size, zoom=zoom)
    return cv2.warpAffine(
        src=image, 
        M=M, 
        dsize=(image_size, image_size), 
        borderValue=0.0), M


class FaceDetection:
    
    def __init__(self):
        self.current_modules = None
        self.model = None

    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input batch of RGB images to detect faces from."}),
                "det_size": ("INT", {"default": 512, "min": 16, "max": 4096, "tooltip": "Size of the cropped face output."}),
                "det_score": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.01, "tooltip": "Minimum confidence score to consider a detection valid."}),
                "min_face_size": ("INT", {"default": 16, "min": 16, "max": 4096, "tooltip": "Ignore faces smaller than this height in pixels."}),
                "zoom": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "Zoom level applied during face alignment."}),
                "interpolation": (["ema", "sma"], {"tooltip": "Smoothing technique for landmark stabilization: EMA (Exponential) or SMA (Simple)."}),
                "ema_alpha": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Smoothing factor for EMA landmark interpolation."}),
                "sma_buffer": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "Number of frames used in SMA landmark interpolation."}),
                "ref_similarity": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum cosine similarity with reference embedding to accept match."}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "Optional reference face image(s) for identity filtering."}),
            }
        }


    RETURN_TYPES = ("IMAGE", "M")
    RETURN_NAMES = ("crops", "M")
    FUNCTION = "detect"
    CATEGORY = "face detection"


    def load_model(self):
        with suppress_output():
            self.model = FaceAnalysis(
                root = INSIGHTFACE_DIR, 
                allowed_modules = self.current_modules,
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            
            self.model.prepare(ctx_id=0, det_size=(640, 480))


    def detect(self, frames, det_size, det_score, min_face_size, zoom, interpolation, ema_alpha, sma_buffer, ref_similarity, reference=None):

        if reference is None:
            allowed_modules = ["detection"]
        else:
            allowed_modules = ["detection", "recognition"]

        if self.current_modules != allowed_modules:
            self.current_modules = allowed_modules
            self.load_model()

        ref_embeds = None
        if reference is not None:
            embeds = []
            for ref in reference:
                ref = ref.detach().cpu().numpy()
                ref = (ref * 255).clip(0, 255).astype(np.uint8)
                ref = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
                faces = self.model.get(ref, max_num=1)
                if faces:
                    embeds.append(faces[0]["embedding"])

            ref_embeds = np.mean(embeds, axis=0)

        pbar = comfy.utils.ProgressBar(frames.shape[0])
        kps = KPS_ema(alpha=ema_alpha) if interpolation == "ema" else KPS_sma(buffer=sma_buffer)

        num_frames = frames.shape[0]
        crops = torch.empty((num_frames, det_size, det_size, 3), dtype=torch.float16)
        tms = []

        for i, frame in enumerate(frames):
            frame = frame.detach().cpu().numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            crop = np.zeros((det_size, det_size, 3), dtype=np.uint8)
            M = None

            faces = self.model.get(frame, max_num=10)

            if faces:
                best_index = 0
                similarity_score = 10

                if ref_embeds is not None:
                    similarity_score = -10
                    for j, face in enumerate(faces):
                        face_height = face["bbox"][3] - face["bbox"][1]
                        if face_height < min_face_size:
                            continue

                        score = cosine_similarity(ref_embeds, face["embedding"])
                        if score > similarity_score:
                            similarity_score = score
                            best_index = j

                face_height = faces[best_index]["bbox"][3] - faces[best_index]["bbox"][1]

                if similarity_score >= ref_similarity and faces[best_index]["det_score"] >= det_score and face_height >= min_face_size:
                    crop, M = norm_crop(frame, kps.update(faces[best_index]["kps"]), image_size=det_size, zoom=zoom)
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crops[i] = torch.from_numpy(crop).to(dtype=torch.float16) / 255.
            tms.append(M)
            pbar.update(1)

        return crops, tms
    

class FaceCombine:
    
    def __init__(self):
        pass

    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Original image frames to combine with cropped faces."}),
                "crops": ("IMAGE", {"tooltip": "Aligned and cropped face images to insert back into frames."}),
                "M": ("M", {"tooltip": "Affine transformation matrices used during cropping, needed for inverse mapping."}),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "Optional masks to blend the face region into the original frame."}),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "combine"
    CATEGORY = "face detection"


    def combine(self, frames, crops, M, masks=None):
        pbar = comfy.utils.ProgressBar(frames.shape[0])

        num_frames, H, W = frames.shape[0], frames.shape[1], frames.shape[2] # (N, H, W, C)
        combined_frames = torch.empty((num_frames, H, W, 3), dtype=torch.float16)

        default_mask = cv2.imread(os.path.join(CURRENT_DIR, "masks/mask1.png"))
        default_mask = cv2.resize(default_mask, crops[0].shape[:2]).astype(np.float32) / 255.
        mask = default_mask

        i = 0
        while i < num_frames:
            frame = frames[i].detach().cpu().numpy().astype(np.float32)
            merged_frame = frame

            if M[i] is not None:
                crop = crops[i].detach().cpu().numpy().astype(np.float32)
                
                if masks is not None:
                    mask = masks[i].detach().cpu().numpy().astype(np.float32)
                    mask = cv2.resize(mask, crops[0].shape[:2])
                    mask = np.stack([mask] * 3, axis=-1)
                    mask = mask * default_mask
                    
                inv_m = cv2.invertAffineTransform(M[i])
                face_image = cv2.warpAffine(src=crop, M=inv_m, dsize=(W, H))
                face_mask  = cv2.warpAffine(src=mask, M=inv_m, dsize=(W, H))

                merged_frame = face_mask * face_image + (1 - face_mask) * frame
            
            combined_frames[i] = torch.from_numpy(merged_frame.astype(np.float16))
            pbar.update(1)
            i += 1

        return (combined_frames,)

    
NODE_CLASS_MAPPINGS = {
    "FaceDetection" : FaceDetection,
    "FaceCombine": FaceCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetection": "FaceDetection",
    "FaceCombine": "FaceCombine",
}