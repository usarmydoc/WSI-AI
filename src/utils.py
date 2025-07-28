"""
General utility functions for image, model, and metrics handling in histopathology AI workflows.

Includes functions for image I/O, visualization, model persistence, metrics logging, and metadata management.

"""

from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os

def load_image(image_path):
    """
    Load an image from the specified path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image: Loaded PIL Image object.
    """
    return Image.open(image_path)

def visualize_image(image, title='Image'):
    """
    Visualize an image with a title using matplotlib.

    Args:
        image (Image): PIL Image or NumPy array.
        title (str): Title for the plot.
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def log_metrics(metrics):
    """
    Log metrics during training or evaluation.

    Args:
        metrics (dict): Dictionary of metric names and values.
    """
    for key, value in metrics.items():
        print(f"{key}: {value}")

def save_model(model, file_path):
    """
    Save the trained model to a file using pickle.

    Args:
        model: Model object to save.
        file_path (str): Path to save the model.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """
    Load a model from a file using pickle.

    Args:
        file_path (str): Path to the model file.

    Returns:
        Loaded model object.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_image(image, file_path):
    """
    Save an image to a specified path.

    Args:
        image (Image): PIL Image object.
        file_path (str): Path to save the image.
    """
    image.save(file_path)

def load_image_from_file(file_path):
    """
    Load an image from a file path.

    Args:
        file_path (str): Path to the image file.

    Returns:
        Image: Loaded PIL Image object.
    """
    return Image.open(file_path)

def save_image_to_file(image, file_path):
    """
    Save an image to a file path.

    Args:
        image (Image): PIL Image object.
        file_path (str): Path to save the image.
    """
    image.save(file_path)

def display_image(image):
    """
    Display an image using matplotlib.

    Args:
        image (Image): PIL Image or NumPy array.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_plot(figure, file_path):
    """
    Save a matplotlib figure to a file.

    Args:
        figure: Matplotlib figure object.
        file_path (str): Path to save the figure.
    """
    figure.savefig(file_path)
    plt.close(figure)

def load_plot(file_path):
    """
    Load a matplotlib figure from a file.

    Args:
        file_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image array.
    """
    return plt.imread(file_path)

def display_plot(figure):
    """
    Display a matplotlib figure.

    Args:
        figure: Matplotlib figure object or image array.
    """
    plt.imshow(figure)
    plt.axis('off')
    plt.show()

def save_metrics(metrics, file_path):
    """
    Save metrics to a file using pickle.

    Args:
        metrics (dict): Dictionary of metrics.
        file_path (str): Path to save the metrics.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(metrics, f)

def load_metrics(file_path):
    """
    Load metrics from a file using pickle.

    Args:
        file_path (str): Path to the metrics file.

    Returns:
        dict: Loaded metrics.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def log_metrics_to_file(metrics, file_path):
    """
    Log metrics to a text file.

    Args:
        metrics (dict): Dictionary of metrics.
        file_path (str): Path to save the log.
    """
    with open(file_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def load_metrics_from_file(file_path):
    """
    Load metrics from a text file.

    Args:
        file_path (str): Path to the metrics file.

    Returns:
        dict: Loaded metrics.
    """
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            metrics[key] = float(value)
    return metrics

def save_image_to_directory(image, directory, filename):
    """
    Save an image to a specified directory with a filename.

    Args:
        image (Image): PIL Image object.
        directory (str): Directory path.
        filename (str): Filename for the image.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    image.save(os.path.join(directory, filename))

def load_image_from_directory(directory, filename):
    """
    Load an image from a specified directory with a filename.

    Args:
        directory (str): Directory path.
        filename (str): Filename for the image.

    Returns:
        Image: Loaded PIL Image object.
    """
    return Image.open(os.path.join(directory, filename))

def display_image_from_directory(directory, filename):
    """
    Display an image from a specified directory with a filename.

    Args:
        directory (str): Directory path.
        filename (str): Filename for the image.
    """
    image = load_image_from_directory(directory, filename)
    display_image(image)

def save_plot_to_directory(figure, directory, filename):
    """
    Save a matplotlib figure to a specified directory with a filename.

    Args:
        figure: Matplotlib figure object.
        directory (str): Directory path.
        filename (str): Filename for the figure.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_plot(figure, os.path.join(directory, filename))

def load_plot_from_directory(directory, filename):
    """
    Load a matplotlib figure from a specified directory with a filename.

    Args:
        directory (str): Directory path.
        filename (str): Filename for the figure.

    Returns:
        np.ndarray: Loaded image array.
    """
    return load_plot(os.path.join(directory, filename))

def display_plot_from_directory(directory, filename):
    """
    Display a matplotlib figure from a specified directory with a filename.

    Args:
        directory (str): Directory path.
        filename (str): Filename for the figure.
    """
    figure = load_plot_from_directory(directory, filename)
    display_plot(figure)

def save_metrics_to_directory(metrics, directory, filename):
    """
    Save metrics to a specified directory with a filename.

    Args:
        metrics (dict): Dictionary of metrics.
        directory (str): Directory path.
        filename (str): Filename for the metrics.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_metrics(metrics, os.path.join(directory, filename))

def log_metrics_to_directory(metrics, directory, filename):
    """
    Log metrics to a specified directory with a filename.

    Args:
        metrics (dict): Dictionary of metrics.
        directory (str): Directory path.
        filename (str): Filename for the log.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_metrics_to_file(metrics, os.path.join(directory, filename))

def load_metrics_from_directory(directory, filename):
    """
    Load metrics from a specified directory with a filename.

    Args:
        directory (str): Directory path.
        filename (str): Filename for the metrics.

    Returns:
        dict: Loaded metrics.
    """
    return load_metrics_from_file(os.path.join(directory, filename))

def save_image_with_metadata(image, metadata, file_path):
    """
    Save an image along with its metadata to a specified file path.
    Metadata is saved in a separate file with the same base name.

    Args:
        image (Image): PIL Image object.
        metadata (dict): Metadata dictionary.
        file_path (str): Path to save the image.
    """
    image.save(file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def load_image_with_metadata(file_path):
    """
    Load an image and its metadata from a specified file path.
    Metadata is expected to be in a separate file with the same base name.

    Args:
        file_path (str): Path to the image file.

    Returns:
        tuple: (Image, metadata dictionary)
    """
    image = Image.open(file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return image, metadata

def save_plot_with_metadata(figure, metadata, file_path):
    """
    Save a matplotlib figure along with its metadata to a specified file path.
    Metadata is saved in a separate file with the same base name.

    Args:
        figure: Matplotlib figure object.
        metadata (dict): Metadata dictionary.
        file_path (str): Path to save the figure.
    """
    save_plot(figure, file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def load_plot_with_metadata(file_path):
    """
    Load a matplotlib figure and its metadata from a specified file path.
    Metadata is expected to be in a separate file with the same base name.

    Args:
        file_path (str): Path to the figure file.

    Returns:
        tuple: (figure image array, metadata dictionary)
    """
    figure = load_plot(file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return figure, metadata

def save_model_with_metadata(model, metadata, file_path):
    """
    Save a model along with its metadata to a specified file path.
    Metadata is saved in a separate file with the same base name.

    Args:
        model: Model object.
        metadata (dict): Metadata dictionary.
        file_path (str): Path to save the model.
    """
    save_model(model, file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def load_model_with_metadata(file_path):
    """
    Load a model and its metadata from a specified file path.
    Metadata is expected to be in a separate file with the same base name.

    Args:
        file_path (str): Path to the model file.

    Returns:
        tuple: (model object, metadata dictionary)
    """
    model = load_model(file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return model, metadata

def save_metrics_with_metadata(metrics, metadata, file_path):
    """
    Save metrics along with its metadata to a specified file path.
    Metadata is saved in a separate file with the same base name.

    Args:
        metrics (dict): Metrics dictionary.
        metadata (dict): Metadata dictionary.
        file_path (str): Path to save the metrics.
    """
    save_metrics(metrics, file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def load_metrics_with_metadata(file_path):
    """
    Load metrics and its metadata from a specified file path.
    Metadata is expected to be in a separate file with the same base name.

    Args:
        file_path (str): Path to the metrics file.

    Returns:
        tuple: (metrics dictionary, metadata dictionary)
    """
    metrics = load_metrics(file_path)
    metadata_file = file_path.rsplit('.', 1)[0] + '_metadata.pkl'
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return metrics, metadata