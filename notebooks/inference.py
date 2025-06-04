import os

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import Config
from src.utils.verification import FaceVerifier


def run_face_verification(demo_dir=".photos", person_name="somebody"):
    """
    Run face verification pipeline

    Args:
        demo_dir: Directory containing demo images
        person_name: Name of the person to verify (subdirectory name)
        num_classes: Number of classes in the trained model

    Returns:
        results: Verification results
        metrics: Performance metrics
    """
    config = Config()

    # Initialize verifier
    verifier = FaceVerifier(config.MODEL_SAVE_PATH, config)

    # Prepare paths
    anchor_path = os.path.join(demo_dir, person_name, f"{person_name}_01.jpg")
    your_images = [os.path.join(demo_dir, person_name, f"{person_name}_0{i}.jpg") for i in range(2, 7)]
    other_images = [os.path.join(demo_dir, "others", f"person_0{i}.jpg") for i in range(1, 11)]
    all_test_images = your_images + other_images

    # Expected results
    expected_labels = [True] * 5 + [False] * 10

    # Verify all images exist
    missing = [img for img in [anchor_path] + all_test_images if not os.path.exists(img)]
    if missing:
        raise FileNotFoundError(f"Missing images: {missing}")

    # Run verification
    results = verifier.verify(anchor_path, all_test_images)

    # Calculate metrics
    accuracy = verifier.evaluate_verification(results, expected_labels)
    tp = sum(1 for r, e in zip(results, expected_labels) if r["is_same"] and e)
    tn = sum(1 for r, e in zip(results, expected_labels) if not r["is_same"] and not e)
    fp = sum(1 for r, e in zip(results, expected_labels) if r["is_same"] and not e)
    fn = sum(1 for r, e in zip(results, expected_labels) if not r["is_same"] and e)

    metrics = {"accuracy": accuracy, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": config.VERIFICATION_THRESHOLD}

    return anchor_path, results, expected_labels, metrics


def visualize_results(anchor_path, results, expected_labels, metrics, person_name):
    """
    Visualize verification results

    Args:
        anchor_path: Path to anchor image
        results: Verification results
        expected_labels: Expected verification labels
        metrics: Performance metrics
        person_name: Name of the person being verified
    """
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Face Verification Results", fontsize=24)

    # Plot anchor image
    ax = plt.subplot(4, 4, 1)
    img = Image.open(anchor_path)
    ax.imshow(img)
    ax.set_title(f"Anchor Image ({person_name})", fontsize=14)
    ax.axis("off")

    # Plot test images with results
    for i, (result, expected) in enumerate(zip(results, expected_labels)):
        ax = plt.subplot(4, 4, i + 2)
        img = Image.open(result["path"])
        ax.imshow(img)

        # Determine result color
        is_correct = result["is_same"] == expected
        color = "green" if is_correct else "red"

        # Create title
        person_type = person_name if i < 5 else "Other"
        title = (
            f"{person_type} {i + 1}\n"
            f"Similarity: {result['similarity']:.3f}\n"
            f"Same: {result['is_same']} ({'✓' if is_correct else '✗'})"
        )

        ax.set_title(title, color=color, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("Face Verification Performance Summary")
    print("=" * 60)
    print(f"Verification Threshold: {metrics['threshold']:.2f}")
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    print("\nDetailed Results:")
    print(f"- True Positives (Correct {person_name}): {metrics['tp']}/5")
    print(f"- True Negatives (Correct OTHERS): {metrics['tn']}/10")
    print(f"- False Positives (Wrong {person_name}): {metrics['fp']}")
    print(f"- False Negatives (Wrong OTHERS): {metrics['fn']}")

    # Print requirement status
    print("\n" + "=" * 60)
    print("Requirement Status:")
    you_req = "PASSED" if metrics["tp"] >= 3 else "FAILED"
    others_req = "PASSED" if metrics["tn"] >= 6 else "FAILED"
    print(f"- Identify {person_name} (≥3/5): {metrics['tp']}/5 → {you_req}")
    print(f"- Reject OTHERS (≥6/10): {metrics['tn']}/10 → {others_req}")

    # Confusion matrix
    y_true = [1 if e else 0 for e in expected_labels]
    y_pred = [1 if r["is_same"] else 0 for r in results]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Other", person_name])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    run_face_verification()
