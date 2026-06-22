import cv2
import numpy as np


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print("❌ Error: Unable to load image.")
    return image


def resize_image(image, width, height):
    return cv2.resize(image, (width, height))


def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]


def apply_filter(image, filter_type):
    if filter_type == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif filter_type == "blur":
        return cv2.GaussianBlur(image, (7, 7), 0)

    elif filter_type == "edges":
        return cv2.Canny(image, 50, 150)

    elif filter_type == "sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return cv2.transform(image, kernel)

    elif filter_type == "invert":
        return cv2.bitwise_not(image)

    else:
        print("❓ Unknown filter type.")
        return image


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return image


def apply_multiple_filters(image):
    print("\nAvailable filters: gray, blur, edges, sepia, invert")

    filters = input(
        "Enter filters separated by spaces (e.g., gray blur): "
    ).split()

    result = image

    for f in filters:
        result = apply_filter(result, f)

    return result


def main():
    path = input("📸 Enter the path to your image: ")

    image = load_image(path)

    if image is None:
        return

    print("\n🔧 Choose an operation:")
    print("1. Resize")
    print("2. Crop")
    print("3. Apply Single Filter")
    print("4. Apply Multiple Filters")
    print("5. Detect Faces")

    choice = input("Your choice (1-5): ")

    if choice == "1":
        width = int(input("New width: "))
        height = int(input("New height: "))
        result = resize_image(image, width, height)

    elif choice == "2":
        x = int(input("X: "))
        y = int(input("Y: "))
        w = int(input("Width: "))
        h = int(input("Height: "))
        result = crop_image(image, x, y, w, h)

    elif choice == "3":
        print("Available filters: gray, blur, edges, sepia, invert")
        f = input("Choose filter: ")
        result = apply_filter(image, f)

    elif choice == "4":
        result = apply_multiple_filters(image)

    elif choice == "5":
        result = detect_faces(image)

    else:
        print("❌ Invalid choice.")
        return

    cv2.imshow("🖼️ Processed Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save = input(
        "💾 Do you want to save the result? (yes/no): "
    ).lower()

    if save == "yes":
        output_path = input(
            "Enter filename to save (e.g., result.jpg): "
        )
        cv2.imwrite(output_path, result)
        print(f"✅ Image saved as {output_path}")


if __name__ == "__main__":
    main()
