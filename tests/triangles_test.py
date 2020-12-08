import matplotlib.pyplot as plot
from PIL import Image
import numpy
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from sklearn.cluster import KMeans
from datetime import date
from reportlab.pdfgen import canvas

from generators.triangles import get_triangles, get_dominant_colour, \
    perlin_noise, make_page, get_days_and_dow, get_centroids, make_page

# points, triangles = get_triangles(8, 6)
# plot.triplot(points[:, 0], points[:, 1], triangles.simplices)
# plot.scatter(points[:, 0], points[:, 1])
# print(triangles.simplices)
# print(triangles.neighbors[2])
# plot.show()

image_path = os.path.join(os.path.dirname(__file__), "..", "img", "sample.jpg")
image = Image.open(image_path)
# print(get_dominant_colour(image, 6))

# x = numpy.linspace(0.0, 1.0, 100)
# seeds = numpy.random.uniform(0.0, 100000.0, [7])
# f = numpy.vectorize(lambda a: perlin_noise(seeds, a, 0.5))
# y = f(x)
# plot.plot(x, y)
# plot.show()

# print(get_days_and_dow(date(2020, 11, 2)))
# get_centroids(points, triangles.simplices)
make_page(date(2020, 11, 12), image)
