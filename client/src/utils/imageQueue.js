export class ImageQueue {
  static instance = null;

  constructor(handleImageUpload) {
    if (ImageQueue.instance) {
      return ImageQueue.instance;
    }

    this.count = 0;
    this.queue = [];
    this.processing = false;
    this.handleImageUpload = handleImageUpload;

    ImageQueue.instance = this;
  }

  static getInstance(handleImageUpload) {
    if (!ImageQueue.instance) {
      ImageQueue.instance = new ImageQueue(handleImageUpload);
    }
    return ImageQueue.instance;
  }

  enqueue(item) {
    this.queue.push(item);
    this.startProcessing();
  }

  async startProcessing() {
    if (this.processing) return;
    this.processing = true;

    while (this.queue.length > 0) {
      const item = this.queue.shift();
      await this.handleImageUpload(item, this.count);
      this.count++;
    }

    this.processing = false;
  }
}
