# Neural Turing Machine

A TensorFlow implementation of the Neural Turing Machine proposed by Graves et al. (https://arxiv.org/abs/1410.5401)

### To Run:
Simply run `python main.py` to run inference on the pretrained model. To run inference on a different trained model, run `python main.py --checkpoint CKPT_DIR`. To retrain the model from scratch, run `python main.py -t --iterations NUM_EPOCHS`.
