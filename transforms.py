import albumentations as A

def get_train_transforms(mean, std, p):

    train_transforms = A.Compose(
        [
            A.Normalize(mean, std),
            A.HorizontalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            A.CoarseDropout(max_holes = 1,
                            max_height=16,
                            max_width=16,
                            min_holes = 1,
                            min_height=16,
                            min_width=16,
                            fill_value=(mean),
                            mask_fill_value = None,
                            p=p
            )
        ]
    )

    return train_transforms


def get_test_transforms(mean, std):
    # Test data transformations
    test_transforms = A.Compose([
        A.Normalize(mean, std)
    ])

    return test_transforms