for category in $(ls ../datasets/mvtec)
do
  python train.py --category $category
done