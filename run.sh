##morph 2-4

python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/3-style.jpg' --output 'examples/origin-13/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-13/output%s.jpg'
python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/4-style.jpg' --output 'examples/origin-14/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-14/output%s.jpg'
python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/5-style.jpg' --output 'examples/origin-15/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-15/output%s.jpg'
python neural_style.py --content 'examples/1-content.jpg' --styles 'examples/6-style.jpg' --output 'examples/origin-16/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-16/output%s.jpg'

python neural_style.py --content 'examples/2-content.jpg' --styles 'examples/3-style.jpg' --output 'examples/origin-23/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-23/output%s.jpg'
python neural_style.py --content 'examples/3-content.jpg' --styles 'examples/4-style.jpg' --output 'examples/origin-34/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-34/output%s.jpg'
python neural_style.py --content 'examples/4-content.jpg' --styles 'examples/5-style.jpg' --output 'examples/origin-45/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-45/output%s.jpg'
python neural_style.py --content 'examples/5-content.jpeg' --styles 'examples/6-style.jpg' --output 'examples/origin-56/output.jpg' --iterations 800 --checkpoint-iterations 50 --checkpoint-output 'examples/origin-56/output%s.jpg'

