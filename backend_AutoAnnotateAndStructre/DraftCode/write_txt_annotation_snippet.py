    def write_txt_annotation(self):
        boxes, labels = self.get_labels_boxes()
        class_id = 0
        annotation_lines = []
        for b in boxes:
            xmin, ymin, xmax, ymax = b
            img_w, img_h = self.image_dims[1], self.image_dims[0]
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            box_w = (xmax - xmin) / img_w
            box_h = (ymax - ymin) / img_h
            annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        output_path = os.path.join(self.annotation_path, self.image_name.rsplit('.', 1)[0] + ".txt")
        with open(output_path, "w") as f:
            f.write("\n".join(annotation_lines))
