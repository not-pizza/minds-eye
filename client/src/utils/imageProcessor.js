export class ImageProcessor {
  static instance = null;

  constructor() {
    if (ImageProcessor.instance) {
      return ImageProcessor.instance;
    }

    this.count = 0;
    this.queue = [];
    this.processing = false;
    this.uploadFunction = null;
    this.onImageProcessed = null;

    ImageProcessor.instance = this;
  }

  static getInstance() {
    if (!ImageProcessor.instance) {
      ImageProcessor.instance = new ImageProcessor();
    }
    return ImageProcessor.instance;
  }

  setUploadFunction(uploadFn) {
    this.uploadFunction = uploadFn;
    return this;
  }

  setOnImageProcessed(callback) {
    this.onImageProcessed = callback;
    return this;
  }

  enqueue(item) {
    this.queue.push(item);
    this.startProcessing();
  }

  async startProcessing() {
    if (this.processing || !this.uploadFunction) return;
    this.processing = true;

    while (this.queue.length > 0) {
      const item = this.queue.shift();
      try {
        const result = await this.uploadFunction(item);
        if (this.onImageProcessed) {
          this.onImageProcessed(result, this.count);
        }
        this.count++;
      } catch (error) {
        console.error("Error processing image:", error);
      }
    }

    this.processing = false;
  }
}