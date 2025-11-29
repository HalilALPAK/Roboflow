
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the saved 'nude.pt' model
model = YOLO('nude.pt')

# Run inference on a new image (replace 'test1.jpeg' with the path to your image)
results = model.predict('test5.jpg')

# Display the results using matplotlib
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    # Convert BGR to RGB for matplotlib
    im_rgb = im_array[..., ::-1]
    plt.imshow(im_rgb)
    plt.axis('off') # Hide axes
    plt.show()