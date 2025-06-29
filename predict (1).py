import argparse
from utils import load_classifier, load_label_map, predict

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model.')
    parser.add_argument('--image_path', type=str, help='Path to input image.')
    parser.add_argument('--model_path', type=str, help='Path to trained Keras model (.h5).')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions.')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file mapping labels to flower names.')
    
    args = parser.parse_args()

    classifier = load_classifier(args.model_path)
    label_map = load_label_map(args.category_names)

    probs, classes = predict(args.image_path, classifier, args.top_k)
    class_names = [label_map[str(c)] for c in classes]

    print("\nTop Predictions:")
    for name, prob in zip(class_names, probs):
        print(f"{name}: {prob:.4f}")

if __name__ == '__main__':
    main()