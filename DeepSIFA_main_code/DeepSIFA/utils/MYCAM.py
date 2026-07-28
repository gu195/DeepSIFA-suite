import cv2
import numpy as np


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers: # attention
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):  # forward propagation process
        self.gradients = [] # Clear previous gradients and activations information
        self.activations = []
        return self.model(x) # Two hook functions will be triggered during forward propagation.

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        # return np.mean(grads, axis=(2, 3), keepdims=True)
        return np.mean(grads, axis=(2), keepdims=True)   # attention averages the height and width of the gradient information

    @staticmethod
    def get_loss(output, target_category):  # len(target_category) The number of pictures in the current batch
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        # res7
        # activations               1, 256, 1024
        # grads                     1, 256, 1024
        # weights                   1, 256, 1
        # weighted_activations      1, 256, 1024
        # cam                       1, 1024
        weights = self.get_cam_weights(grads)   # error reported here
        # np.savetxt('weights.txt', weights[0], fmt='%f')
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        # np.savetxt('cam.txt', cam[0], fmt='%f', newline='\n')
        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return height, width

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()    # Save activations to activations_list
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()  # Save gradients to grads_list
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)    # Get the height and width of the input image

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            # cam   1 1024
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image Relu
            # np.savetxt('camcam<0.txt', cam[0], fmt='%f', newline='\n')
            scaled = self.scale_cam_image(cam, target_size) # scaled    Batch 1 1024
            # np.savetxt('scaled.txt', scaled[0][0], fmt='%f', newline='\n')
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            # np.savetxt('img.txt', img, fmt='%f', newline='\n')
            if target_size is not None:
                img = cv2.resize(img, target_size)  # Resize cam to the size of the original image
                img = np.reshape(img, target_size)
            result.append(img)  # img becomes 1 1024
        result = np.float32(result) # result becomes 1 1 1024

        return result   

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # forward propagation to obtain network output logits (without softmax)
        output = self.activations_and_grads(input_tensor)   # An error is reported at this step. Why?
        # output = output[0]
        if isinstance(target_category, int):    # Find gradcam of multiple pictures at one time
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None: # Which one generates the corresponding heat map?
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()  # Clear the historical gradient information of the model
        loss = self.get_loss(output, target_category) #
        loss.backward(retain_graph=True) # triggers the hook function and captures gradient information

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)    # There is an error in calculating the heat map based on the formula.
        return self.aggregate_multi_layers(cam_per_layer)   

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap) # 1 1024 becomes 1 1024 3
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255 # 1 1024 3

    if np.max(img) > 1:# img 1 1024
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    # Add channel dimension to img
    img = np.expand_dims(img, axis=-1)  # 1 1024 1
    cam = heatmap + img # 1 1024 3
    cam = cam / np.max(cam) # 1 1024 3
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img
