def bb_file(patient,leg):
    return f"{patient}_export2_{leg}_image_ext2_cropped.pkl"
    
def mask_file(patient,leg):
    return f"{patient}_export2_{leg}_labelmask.nii.gz"

def femur_file(patient,leg):
    return f"{patient}_export2_{leg}_image_ext2_cropped.nii.gz"