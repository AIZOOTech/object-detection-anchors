import glob
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, avg_iou

# ANNOTATIONS_PATH = "./data/pascalvoc07-annotations"
ANNOTATIONS_PATH = "./data/widerface-annotations"
CLUSTERS = 25
BBOX_NORMALIZE = True

def show_cluster(data, cluster, max_points=2000):
	'''
	Display bouding box's size distribution and anchor generated in scatter.
	'''
	if len(data) > max_points:
		idx = np.random.choice(len(data), max_points)
		data = data[idx]
	plt.scatter(data[:,0], data[:,1], s=5, c='lavender')
	plt.scatter(cluster[:,0], cluster[:, 1], c='red', s=100, marker="^")
	plt.xlabel("Width")
	plt.ylabel("Height")
	plt.title("Bounding and anchor distribution")
	plt.savefig("cluster.png")
	plt.show()

def show_width_height(data, cluster, bins=50):
	'''
	Display bouding box distribution with histgram.
	'''
	if data.dtype != np.float32:
		data = data.astype(np.float32)
	width = data[:, 0]
	height = data[:, 1]
	ratio = height / width

	plt.figure(1,figsize=(20, 6))
	plt.subplot(131)
	plt.hist(width, bins=bins, color='green')
	plt.xlabel('width')
	plt.ylabel('number')
	plt.title('Distribution of Width')

	plt.subplot(132)
	plt.hist(height,bins=bins, color='blue')
	plt.xlabel('Height')
	plt.ylabel('Number')
	plt.title('Distribution of Height')

	plt.subplot(133)
	plt.hist(ratio, bins=bins,  color='magenta')
	plt.xlabel('Height / Width')
	plt.ylabel('number')
	plt.title('Distribution of aspect ratio(Height / Width)')
	plt.savefig("shape-distribution.png")
	plt.show()
	

def sort_cluster(cluster):
	'''
	Sort the cluster to with area small to big.
	'''
	if cluster.dtype != np.float32:
		cluster = cluster.astype(np.float32)
	area = cluster[:, 0] * cluster[:, 1]
	cluster = cluster[area.argsort()]
	ratio = cluster[:,1:2] / cluster[:, 0:1]
	return np.concatenate([cluster, ratio], axis=-1)


def load_dataset(path, normalized=True):
	'''
	load dataset from pasvoc formatl xml files
	'''
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			if normalized:
				xmin = int(obj.findtext("bndbox/xmin")) / float(width)
				ymin = int(obj.findtext("bndbox/ymin")) / float(height)
				xmax = int(obj.findtext("bndbox/xmax")) / float(width)
				ymax = int(obj.findtext("bndbox/ymax")) / float(height)
			else:
				xmin = int(obj.findtext("bndbox/xmin")) 
				ymin = int(obj.findtext("bndbox/ymin")) 
				xmax = int(obj.findtext("bndbox/xmax")) 
				ymax = int(obj.findtext("bndbox/ymax"))
			if (xmax - xmin) == 0 or (ymax - ymin) == 0:
				continue # to avoid divded by zero error.
			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

print("Start to load data annotations on: %s" % ANNOTATIONS_PATH)
data = load_dataset(ANNOTATIONS_PATH, normalized=BBOX_NORMALIZE)

print("Start to do kmeans, please wait for a moment.")
out = kmeans(data, k=CLUSTERS)

out_sorted = sort_cluster(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

show_cluster(data, out, max_points=2000)

if out.dtype != np.float32:
	out = out.astype(np.float32)

print("Recommanded aspect ratios(width/height)")
print("Width    Height   Height/Width")
for i in range(len(out_sorted)):
	print("%.3f      %.3f     %.1f" % (out_sorted[i,0], out_sorted[i,1], out_sorted[i,2]))
show_width_height(data, out, bins=50)
