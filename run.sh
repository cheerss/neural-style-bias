mkdir examples/0606

CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/tubingen.jpg' --styles 'examples/1-style.jpg' --iterations 150 --output 'examples/0606/1.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/4-content.jpg' --styles 'examples/2-style1.jpg' --iterations 150 --output 'examples/0606/2.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/7-style.jpg' --iterations 150 --output 'examples/0606/7.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/8-style.jpg' --iterations 150 --output 'examples/0606/8.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/3-style.jpg' --iterations 150 --output 'examples/0606/3.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/4-style.jpg' --iterations 150 --output 'examples/0606/4.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/5-style.jpg' --iterations 150 --output 'examples/0606/5.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/9-style.jpg' --iterations 150 --output 'examples/0606/9.jpg'

