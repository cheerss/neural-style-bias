mkdir examples/0527

CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/4-content.jpg' --styles 'examples/5-style.jpg' --iterations 800 --output 'examples/0527/45.jpg'
CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/fran.jpeg' --styles 'examples/6-style.jpg' --iterations 800 --output 'examples/0527/f6.jpg'
CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/2-content.jpg' --styles 'examples/3-style.jpg' --iterations 800 --output 'examples/0527/23.jpg'

CUDA_VISIBLE_DEVICES=2 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/4-style.jpg' --iterations 800 --output 'examples/0527/14.jpg'
