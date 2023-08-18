import numpy as np
from nipy.utils import example_data
from nipy.modalities.fmri.glm import FMRILinearModel
fmri_files = [example_data.get_filename('fiac', 'fiac0', run)
    for run in ['run1.nii.gz', 'run2.nii.gz']]
design_files = [example_data.get_filename('fiac', 'fiac0', run)
    for run in ['run1_design.npz', 'run2_design.npz']]
mask = example_data.get_filename('fiac', 'fiac0', 'mask.nii.gz')
multi_session_model = FMRILinearModel(fmri_files,
                                      design_files,
                                      mask)
multi_session_model.fit()
z_image, = multi_session_model.contrast([np.eye(13)[1]] * 2)

# The number of voxels with p < 0.001 given by ...
print(np.sum(z_image.get_data() > 3.09))