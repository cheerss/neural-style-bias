

CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/tubingen.jpg' --styles 'examples/sumiao.jpg' --iterations 150 --output 'examples/output-1.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/4-content.jpg' --styles 'examples/sumiao.jpg' --iterations 150 --output 'examples/output-2.jpg'
CUDA_VISIBLE_DEVICES=0 python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/sumiao.jpg' --iterations 150 --output 'examples/output-3.jpg'
