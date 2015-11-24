from PIL import Image, ImageChops

#http://stackoverflow.com/questions/19271692/removing-borders-from-an-image-in-python
im = Image.open('output/foo.png')

def trim(im):
	bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
   	diff = ImageChops.difference(im, bg)
   	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
   	if bbox:
   		return im.crop(bbox)

trim(im).show()