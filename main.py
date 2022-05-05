import ResImageNet

def main():
    model = ResImageNet(try_to_train=True, image_dir="Images", epochs=10)
    model.testImage()
    return

if __name__ == "__main__":
    main()