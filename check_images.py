import json

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

adversarial = load_jsonl("/research/hal-afsharim/learn-to-steer/data/coco/coco_data/coco_pope_adversarial.json")
popular     = load_jsonl("/research/hal-afsharim/learn-to-steer/data/coco/coco_data/coco_pope_popular.json")
random      = load_jsonl("/research/hal-afsharim/learn-to-steer/data/coco/coco_data/coco_pope_random.json")

def get_images(data):
    return set(x.get('image', x.get('image_id')) for x in data)

adv_images = get_images(adversarial)
pop_images = get_images(popular)
rand_images = get_images(random)

print(f"Number of total items in adversarial: {len(adversarial)}")
print(f"Number of total items in popular: {len(popular)}")
print(f"Number of total items in random: {len(random)}")

print(f"Number of unique images in adversarial: {len(adv_images)}")
print(f"Number of unique images in popular: {len(pop_images)}")
print(f"Number of unique images in random: {len(rand_images)}")

print(f"Intersection of all three sets of images: {len(adv_images.intersection(pop_images, rand_images))}")
