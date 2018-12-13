"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
process_root = "data/processed"
batch_size = 128

# params for source dataset
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.99
