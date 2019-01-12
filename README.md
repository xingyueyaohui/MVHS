# MVHS

This the project for human image synthesize. We have conducted exploration on Market1501 and our method shows competitive performance.
Our method includes:
* Using geometric transformations according to human pose to generate original result images.
* Using PNGAN and UNet for refinement.
* Using a shallow fully connected network(poseNet) to select better parts.
* Combine the refined images into a final one.
