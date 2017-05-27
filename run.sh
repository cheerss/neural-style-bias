mkdir examples/0527

CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/tubingen.jpg' --styles 'examples/1-style.jpg' --iterations 800 --output 'examples/0527/t1.jpg'
CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/tubingen.jpg' --styles 'examples/7-style.jpg' --iterations 800 --output 'examples/0527/t7.jpg'
CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/tubingen.jpg' --styles 'examples/8-style.jpg' --iterations 800 --output 'examples/0527/t8.jpg'

CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/tubingen.jpg' --styles 'examples/9-style.jpg' --iterations 800 --output 'examples/0527/t9.jpg'
