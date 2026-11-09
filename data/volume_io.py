"""Step 1 - Volumetric parsing: NIfTI + DICOM readers, DICOM->NIfTI, dataset catalogue."""
from pathlib import Path

import numpy as np
import nibabel as nib


def load_nifti(path):
    """Return (array, affine, spacing) for a NIfTI file."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine, img.header.get_zooms()[:3]


def save_nifti(array, affine, path):
    """Save with the ORIGINAL affine so masks line up with the scan."""
    nib.save(nib.Nifti1Image(array, affine), str(path))


def load_dicom_series(folder):
    """Read a folder of .dcm slices -> (float32 array [rows, cols, slices], RAS affine, spacing).

    Slices are sorted by their position along the slice normal; rescale slope/intercept
    are applied (so CT comes out in Hounsfield units).
    """
    import pydicom
    files = []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and not p.name.startswith("."):
            try:
                ds = pydicom.dcmread(str(p))
            except Exception:
                continue
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelData"):
                files.append(ds)
    if len(files) < 2:
        raise ValueError(f"no DICOM series found in {folder}")

    iop = np.array(files[0].ImageOrientationPatient, dtype=float)
    xvec, yvec = iop[:3], iop[3:]               # direction of increasing column / row index
    normal = np.cross(xvec, yvec)
    files.sort(key=lambda d: float(np.dot(np.array(d.ImagePositionPatient, dtype=float), normal)))

    slices = []
    for d in files:
        arr = d.pixel_array.astype(np.float32)
        arr = arr * float(getattr(d, "RescaleSlope", 1.0)) + float(getattr(d, "RescaleIntercept", 0.0))
        slices.append(arr)
    vol = np.stack(slices, axis=-1)             # [rows, cols, slices]

    p0 = np.array(files[0].ImagePositionPatient, dtype=float)
    p1 = np.array(files[-1].ImagePositionPatient, dtype=float)
    step = (p1 - p0) / (len(files) - 1)         # vector between consecutive slices
    dy, dx = (float(v) for v in files[0].PixelSpacing)   # row spacing, column spacing
    aff = np.eye(4)
    aff[:3, 0] = yvec * dy                      # axis 0 = row index
    aff[:3, 1] = xvec * dx                      # axis 1 = column index
    aff[:3, 2] = step
    aff[:3, 3] = p0
    aff = np.diag([-1, -1, 1, 1]) @ aff         # DICOM is LPS -> NIfTI is RAS
    spacing = (dy, dx, float(np.linalg.norm(step)))
    return vol, aff, spacing


def dicom_to_nifti(folder, out_path):
    vol, aff, _ = load_dicom_series(folder)
    nib.save(nib.Nifti1Image(vol, aff), str(out_path))
    return out_path


def build_catalogue(images_dir, labels_dir=None):
    """List every case with shape / spacing / orientation (useful for EDA). Returns list of dicts."""
    rows = []
    for p in sorted(Path(images_dir).glob("*.nii*")):
        if p.name.startswith("._"):
            continue
        img = nib.load(str(p))
        row = {"id": p.name.split(".")[0], "image": str(p), "shape": tuple(img.shape),
               "spacing": tuple(round(float(s), 3) for s in img.header.get_zooms()[:3]),
               "orientation": "".join(nib.aff2axcodes(img.affine))}
        if labels_dir is not None:
            lp = Path(labels_dir) / p.name
            row["label"] = str(lp) if lp.exists() else None
        rows.append(row)
    return rows
