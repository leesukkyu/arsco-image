const Jimp = require('jimp');
const cv = require('../lib/opencv.js');

async function init(){
      // load local image file with jimp. It supports jpg, png, bmp, tiff and gif:
  var jimpSrc = await Jimp.read('/Users/user/Desktop/git/arsco-image/node/src/lena.jpg');
  // `jimpImage.bitmap` property has the decoded ImageData that we can use to create a cv:Mat
  var src = cv.matFromImageData(jimpSrc.bitmap);
  // following lines is copy&paste of opencv.js dilate tutorial:
  let dst = new cv.Mat();
  let M = cv.Mat.ones(5, 5, cv.CV_8U);
  let anchor = new cv.Point(-1, -1);
  cv.dilate(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
  // Now that we are finish, we want to write `dst` to file `output.png`. For this we create a `Jimp`
  // image which accepts the image data as a [`Buffer`](https://nodejs.org/docs/latest-v10.x/api/buffer.html).
  // `write('output.png')` will write it to disk and Jimp infers the output format from given file name:
  new Jimp({
    width: dst.cols,
    height: dst.rows,
    data: Buffer.from(dst.data)
  })
  .write('output.png');
  src.delete();
  dst.delete();
}

function init2(){
    backSub = cv.BackgroundSubtractorMOG2()
    console.log(backSub)
}

init2()