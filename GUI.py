import tkinter as tk
from PIL import Image, ImageDraw

from main import Main


# Class for the gui
# Contains all elements for the user in and output
class GUI(tk.Tk):

    # Constructor initialised all the gui elemments
    def __init__(self, main):
        super().__init__()

        # Create the Canvas
        self.canvas = tk.Canvas(self, width=600, height=600, bg="black")
        self.canvas.grid(row=0, column=0, rowspan=2)
        self.main = main

        self.canvas.bind("<B1-Motion>", self.paint)

        # Create the Text widget
        self.text_widget = tk.Text(self)
        self.text_widget.grid(row=2, column=0, columnspan=1)
        self.text_widget.insert(tk.END, "Train Model to start...")


        # Create Button frame and buttons
        button_frame = tk.Frame(self)
        button_frame.grid(row=0, column=1)
        tk.Button(button_frame, text="Run", command=self.convert_to_image).pack()
        tk.Button(button_frame, text="Clear Canvas", command=self.clearCanvas).pack()
        tk.Button(button_frame, text="Open Webcam", command=self.camera).pack()
        tk.Button(button_frame, text="Load Model", command=self.main.loadModel).pack()
        tk.Button(button_frame, text="Performance", command=self.getPerformance).pack()
        tk.Button(button_frame, text="Train Model", command=self.trainMain).pack()
        pass

    # Method to draw on Canvas using mouse events
    def paint(self, event):
        x1, y1 = (event.x - 25), (event.y - 25)
        x2, y2 = (event.x + 25), (event.y + 25)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", width=25)
        pass

    # Method to clear the Canvas
    def clearCanvas(self):
        self.canvas.delete("all")
        pass

    # Method to train the neural network from the main components
    def trainMain(self):
        self.main.trainModel(self.text_widget)
        pass

    # Method to run a webcam picture trough the neural network
    def camera(self):
        self.main.openCamera(self.text_widget)
        pass

    # Method to test the accuracy of the neural network
    def getPerformance(self):
        self.main.performance(self.text_widget)
        pass

    # Method to convert the canvas to an image
    def convert_to_image(self):
        
        # Prepare image data
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(image)

        # Iterate over all items in the canvas and draw them on the image
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:
                draw.ellipse(coords, fill="white")
                pass
            else:
                draw.line(coords, fill="white", width=10)
                pass
            pass

        # Scale and save the image for further use
        image = image.resize((28, 28), Image.NEAREST)
        image.save("assets/image.png")

        # Runs the drawing trough the neural network for a prediction
        self.main.runDrawing(self.text_widget)
        pass
    pass


# main method
if __name__ == "__main__":

    # Create architecture of the neural network
    input_nodes = 784
    hidden_nodes = 250
    output_nodes = 10
    learning_rate = 0.05

    main = Main(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Run gui
    app = GUI(main)
    app.mainloop()
    pass