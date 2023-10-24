The repo uses RNN for Image Classification on MNIST Dataset

Model Accuracy : **95.26 %** (Epochs = 10)

**SALIENT POINTS**
  1. **Effective for Classification:** Recurrent Neural Networks (RNNs) are well-suited for classification tasks. They have the capability to model sequential data, making them a suitable choice for tasks where the order and context of data are crucial.
  2. **Input Shape:** In the context of image classification, an RNN can be employed to process image data as a sequence. For example, when working with images of size (28, 28), the RNN takes these images and treats them as a sequence of 28 rows (sequence length) with each row having an input size of 28, corresponding to the image's width.
  3. **Output Generation:** To produce a classification result, the RNN processes the input sequence and generates an output sequence. However, in many classification tasks, we are primarily interested in the final prediction. In such cases, the last output of the RNN is often extracted and used for further processing.
  4. **Classification Decision:** The last RNN output is typically fed into a linear layer, which produces a set of logits. These logits represent the model's confidence scores for different classes. The class corresponding to the highest logit value, i.e., the index associated with the maximum value among the logits, is considered the final output or classification decision.
