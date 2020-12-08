import os
from typing import Tuple, List
from PIL import Image
import numpy
from numpy import cos, sin
from reportlab.lib import colors
from reportlab.platypus import Table
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from datetime import date
from calendar import Calendar
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4, inch
from itertools import zip_longest


def get_triangles(x_seg: int, y_seg: int) -> Tuple[numpy.ndarray, Delaunay]:
    y_coords, x_coords = numpy.meshgrid(range(y_seg + 1), range(x_seg + 1))
    x_coords = x_coords.astype(numpy.float64)
    y_coords = y_coords.astype(numpy.float64)
    x_noise = numpy.random.uniform(-0.3, 0.3, [x_seg - 1, y_seg - 1])
    y_noise = numpy.random.uniform(-0.3, 0.3, [x_seg - 1, y_seg - 1])
    x_coords[1:-1, 1:-1] += x_noise
    y_coords[1:-1, 1:-1] += y_noise
    points = numpy.array([x_coords / x_seg, y_coords / y_seg])
    points = numpy.reshape(points, [2, (x_seg + 1) * (y_seg + 1)])
    points = numpy.transpose(points)
    triangles = Delaunay(points)
    return points, triangles


def get_centroids(points: numpy.ndarray, triangles: numpy.ndarray) -> numpy.ndarray:
    triangle_points = points[triangles]
    point_count = triangle_points.shape[0]
    centroids = numpy.zeros([2, point_count])
    for i in range(point_count):
        centroids[:, i] = numpy.transpose(triangle_points[i]).dot(numpy.array([1/3, 1/3, 1/3]))
    return centroids


def get_dominant_colour(im: Image, steps: int) -> numpy.ndarray:
    image = im.copy()
    image.thumbnail((128, 128))
    rgb_image = image.convert("RGB")
    colours = numpy.array(rgb_image)
    colours = colours.reshape((colours.shape[0] * colours.shape[1], 3))
    model = KMeans(n_clusters=steps).fit(colours)
    dominant = model.cluster_centers_
    brightness = -1 * numpy.matmul(dominant, numpy.array([0.21, 0.72, 0.07]))
    sorted_index = numpy.argsort(brightness)
    dominant_colours = dominant[sorted_index]
    return dominant_colours


def perlin_noise(seeds: numpy.ndarray, x: float, y: float) -> float:
    f = lambda ix, iy: seeds[0] * sin(seeds[1] * ix + seeds[2] * iy + seeds[3])\
                       * cos(seeds[4] * ix + seeds[5] * iy + seeds[6])
    v1 = numpy.array([cos(f(0, 0)), sin(f(0, 0))])
    v2 = numpy.array([cos(f(1, 0)), sin(f(1, 0))])
    v3 = numpy.array([cos(f(1, 1)), sin(f(1, 1))])
    v4 = numpy.array([cos(f(0, 1)), sin(f(0, 1))])
    v = numpy.array([x, y])
    d1 = (v - numpy.array([0, 0])).dot(v1)
    d2 = (v - numpy.array([1, 0])).dot(v2)
    d3 = (v - numpy.array([1, 1])).dot(v3)
    d4 = (v - numpy.array([0, 1])).dot(v4)
    bottom = (1 - x) * d1 + x * d2
    top = (1 - x) * d4 + x * d3
    interpolated = (1 - y) * bottom + y * top
    return max(min(interpolated + 0.5, 0.9999), 0.0)


def get_colour(values: numpy.ndarray, colours: numpy.ndarray) -> numpy.ndarray:
    steps = colours.shape[0]
    colour_values = numpy.floor(values * steps).astype(numpy.int8)
    result = colours[colour_values]
    return result


def get_days_and_dow(year_and_month: date) -> List:
    calendar = Calendar()
    days_and_dow = calendar.itermonthdays2(year_and_month.year, year_and_month.month)
    dow_text = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days_and_dow_text = [(dow_text[dd[1]], dd[0]) for dd in days_and_dow if not dd[0] == 0]
    return days_and_dow_text


def make_page(c: Canvas, year_and_month: date, image: Image):
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    points, triangles = get_triangles(50, 25)
    centroids = get_centroids(points, triangles.simplices)
    dominants = get_dominant_colour(image, 12)
    seeds = numpy.random.uniform(0.0, 100000.0, [7])
    values = numpy.zeros([centroids.shape[1]])
    for p in range(centroids.shape[1]):
        values[p] = perlin_noise(seeds, centroids[0, p], centroids[1, p])
    bright_seeds = numpy.random.uniform(-16, 16, [centroids.shape[1]])
    colours = get_colour(values, dominants) + numpy.transpose(numpy.array([bright_seeds, bright_seeds, bright_seeds]))
    colours = numpy.clip(colours, 0, 255)

    width, height = A4
    c.translate(0.5 * inch, 0.5 * inch + height * 2 / 3)
    pw, ph = width - 1 * inch, height / 3 - 1 * inch
    for i, t in enumerate(triangles.simplices):
        vertices = points[t]
        colour = colours[i]
        c.setStrokeColorRGB(colour[0] / 255, colour[1] / 255, colour[2] / 255)
        c.setFillColorRGB(colour[0] / 255, colour[1] / 255, colour[2] / 255)
        c.setLineJoin(1)
        path = c.beginPath()
        path.moveTo(vertices[0, 0] * pw, vertices[0, 1] * ph)
        path.lineTo(vertices[1, 0] * pw, vertices[1, 1] * ph)
        path.lineTo(vertices[2, 0] * pw, vertices[2, 1] * ph)
        path.close()
        c.drawPath(path, stroke=1, fill=1)

    c.translate(0, -1 * inch)
    c.setFont("Helvetica-Bold", 24)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(0, 0, months[year_and_month.month - 1])

    c.translate(0, -0.5 * inch)
    data = [(d[0], d[1], "") for d in get_days_and_dow(year_and_month)]
    if len(data) < 32:
        for _ in range(32 - len(data)):
            data.append(("", "", ""))
    data = zip_longest(data[:16], data[16:], fillvalue="")
    data = [[dt[0][0], dt[0][1], dt[0][2], dt[1][0], dt[1][1], dt[1][2]] for dt in data]
    table = Table(data, 2 * [0.5 * inch, 0.5 * inch, 2.63 * inch], 16 * [0.32 * inch])
    style = [('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ('GRID', (0, 0), (-1, -1), 1, colors.black), ('BOX', (0, 0), (-1, -1), 2, colors.black)]
    for row_no, row in enumerate(data):
        if row[0] == 'Sun' or row[0] == 'Sat':
            style.append(('TEXTCOLOR', (0, row_no), (2, row_no), colors.green))
        else:
            style.append(('TEXTCOLOR', (0, row_no), (2, row_no), colors.black))
        if row[3] == 'Sun' or row[3] == 'Sat':
            style.append(('TEXTCOLOR', (3, row_no), (-1, row_no), colors.green))
        else:
            style.append(('TEXTCOLOR', (3, row_no), (-1, row_no), colors.black))
    table.setStyle(style)
    aw, ah = width - 1 * inch, height * 2 / 3 - 1.5 * inch
    w, h = table.wrap(aw, ah)
    c.translate(0, -h)
    table.drawOn(c, 0, 0)
    c.showPage()


def make(year: int, images: [Image]):
    c = Canvas("sample.pdf", pagesize=A4)
    for month in range(min(len(images), 12)):
        year_and_month = date(year, month + 1, 1)
        make_page(c, year_and_month, images[month])
    c.save()


if __name__ == '__main__':
    images = [
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "01.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "02.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "03.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "04.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "05.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "06.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "07.jpg")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "08.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "09.jpg")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "10.jpg")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "11.png")),
        Image.open(os.path.join(os.path.dirname(__file__), "..", "img", "12.jpg"))
    ]
    make(2021, images)

