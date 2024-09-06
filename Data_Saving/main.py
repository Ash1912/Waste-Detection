class InMemoryImageStore:
    def __init__(self):
        self.images = []
        print("Initialized in-memory image store.")

    def save_image(self, image):
        """ Save a processed image in memory. """
        self.images.append(image)
        print("Image saved in-memory.")

    def get_images(self):
        """ Retrieve all images stored in memory. """
        return self.images
