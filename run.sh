##morph 2-4

python neural_style.py --content 'examples/2-content.jpg' --styles 'examples/3-style.jpg' --output 'examples/bias_loss-23/output.jpg' --iterations 250 --checkpoint-iterations 50 --checkpoint-output 'examples/bias_loss-23/output%s.jpg'
python neural_style.py --content 'examples/3-content.jpg' --styles 'examples/4-style.jpg' --output 'examples/bias_loss-34/output.jpg' --iterations 250 --checkpoint-iterations 50 --checkpoint-output 'examples/bias_loss-34/output%s.jpg'
python neural_style.py --content 'examples/4-content.jpg' --styles 'examples/5-style.jpg' --output 'examples/bias_loss-45/output.jpg' --iterations 250 --checkpoint-iterations 50 --checkpoint-output 'examples/bias_loss-45/output%s.jpg'
python neural_style.py --content 'examples/5-content.jpeg' --styles 'examples/6-style.jpg' --output 'examples/bias_loss-56/output.jpg' --iterations 250 --checkpoint-iterations 50 --checkpoint-output 'examples/bias_loss-56/output%s.jpg'

