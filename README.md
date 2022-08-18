# SCI_IQA
Folder program-2020 is unavailable.

ATS-CNN's codes are in Folder ATS-CNN_SIQAD.(ATS-CNN: Asymmetric two-stream network for screen content image quality assessment based on region features)

## getTFfile_SIQAD_unite_random.py
getTFfile_SIQAD_unite_random.py converts image data into tfrecord files.
All path values in this file need to be reconfigured carefully.

    run getTFfile_SIQAD_unite_random.py

## mytwo_unite_san_cha_random.py
mytwo_unite_san_cha_random.py builds the CNN network framework and trains the network. The path value in this file need to be reconfigured.

Also, SIQAD_ready_unite_random.py needs to be placed in the same folder as mytwo_unite_san_cha_random.py.Because SIQAD_ready_unite_random.py contains some parameters and functions necessary to train the model.

    run SIQAD_ready_unite_random.py

## mytwo_test_all.py
mytwo_test_all.py is the test file. The data of the trained network model is required when the file is run.In addition, mytwo_unite_san_cha_random.py can become mytwo_test_all.py after some code changes.
