/*
* Copyright 2021 Axel Waggershauser
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package com.example.zxingcpp

import android.graphics.*
import androidx.camera.core.ImageProxy


class BarcodeReader {


    private var sWidth //width
            = 0
    private var sHeight //height
            = 0
    private var sRow //Row--height
            = 0
    private var sCol //col--width
            = 0
    private var sPixel = 0
    private var sIndex = 0

    //ARGB values
    private var sA = 0
    private var sR = 0
    private var sG = 0
    private var sB = 0

    private lateinit var sPixels: IntArray


    // Enumerates barcode formats known to this package.
    // Note that this has to be kept synchronized with native (C++/JNI) side.
    enum class Format {
        NONE, AZTEC, CODABAR, CODE_39, CODE_93, CODE_128, DATA_BAR, DATA_BAR_EXPANDED,
        DATA_MATRIX, EAN_8, EAN_13, ITF, MAXICODE, PDF_417, QR_CODE, UPC_A, UPC_E,
    }

    data class Options(
        val formats: Set<Format> = setOf(),
        val tryHarder: Boolean = false,
        val tryRotate: Boolean = false
    )

    data class Result(
        val format: Format = Format.NONE,
        val text: String? = null,
        val time: String? = null, // for development/debug purposes only
    )

    private lateinit var bitmapBuffer: Bitmap
    var options : Options = Options()

    fun read(image: ImageProxy, invert:Boolean = false): Result? {
        if (!::bitmapBuffer.isInitialized || bitmapBuffer.width != image.width || bitmapBuffer.height != image.height) {
            if (image.format != ImageFormat.YUV_420_888) {
                error("invalid image format")
            }
            bitmapBuffer = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ALPHA_8)
        }
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

//        if(invert)
//            bitmapBuffer = invertBitmap(bitmapBuffer)
        return read(bitmapBuffer, image.cropRect, image.imageInfo.rotationDegrees)
    }

    /**
     * The picture is binarized black and white
     */
    fun zeroAndOne(bm: Bitmap): Bitmap {
        val width = bm.width //The original image width
        val height = bm.height //Original image height
        var color: Int //Used to store the color value of a pixel
        var r: Int
        var g: Int
        var b: Int
        var a: Int //red, green, blue, transparency
        //Create a blank image, the width is equal to the width of the original image, and the height is equal to the height of the original image. Use ARGB_8888 to render. You don’t need to understand this, just write it like this
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val oldPx = IntArray(width * height) //used to store the color information of each pixel in the original image
        val newPx = IntArray(width * height) //Used to process the color information of each pixel after processing
        /**
         * The first parameter oldPix[]: the array used to receive (store) the color information of the pixels in the bm image
         * The second parameter offset: the subscript value of the first receiving color information in the oldPix[] array
         * The third parameter width: the number of items skipping pixels between lines, must be greater than or equal to the number of pixels in each line of the image
         * The fourth parameter x: the abscissa of the first pixel read from the image bm
         * The fifth parameter y: the ordinate of the first pixel read from the image bm
         * The sixth parameter width: the number of pixels that need to be read per line
         * The seventh parameter height: the total number of rows to be read
         */
        bm.getPixels(oldPx, 0, width, 0, 0, width, height) //Get the pixel information in the original image
        for (i in 0 until width * height) { //Process the color value of each pixel in the image in a loop
            color = oldPx[i] //Get the pixel value of a certain point
            r = Color.red(color) //Get the r (red) component of this pixel
            g = Color.green(color) //Get the g (green) component of this pixel
            b = Color.blue(color) //Get the b (blue component) of this pixel
            a = Color.alpha(color) //Get the a channel value of this pixel

            //This formula calculates r, g, b to obtain the gray value, the empirical formula does not need to be understood
            var gray = (r.toFloat() * 0.3 + g.toFloat() * 0.59 + b.toFloat() * 0.11).toInt()
            //The first two ifs below are used for overflow processing to prevent the grayscale formula from getting grayscale beyond the range (0-255)
            if (gray > 255) {
                gray = 255
            }
            if (gray < 130) {
                gray = 0
            }
            if (gray != 0) { //If the gray value of a pixel is not 0 (black), set it to 255 (white)
                gray = 255
            }

            gray = if(gray == 0) {
                255
            } else {
                0;
            }
            newPx[i] = Color.argb(a, gray, gray, gray) //Recompose the processed transparency (unchanged), r, g, b components into color values ​​and store them in the array
        }
        /**
         * The first parameter newPix[]: the color array that needs to be assigned to the new image //The colors to write the bitmap
         * The second parameter offset: the first one in the newPix[] array that needs to be set to the subscript value of the image color //The index of the first color to read from pixels[]
         * The third parameter width: the number of items that skip pixels between rows//The number of colors in pixels[] to skip between rows.
         * Normally this value will be the same as the width of the bitmap,but it can be larger(or negative).
         * The fourth parameter x: the abscissa of the first pixel read from the image bm //The x coordinate of the first pixels to write to in the bitmap.
         * The fifth parameter y: The y coordinate of the first pixels to write to in the bitmap.
         * The sixth parameter width: The number of colors to copy from pixels[] per row.
         * The seventh parameter height: the total number of rows to be read//The number of rows to write to the bitmap.
         */
        bmp.setPixels(newPx, 0, width, 0, 0, width, height) //Assign the processed pixel information to the new image
        return bmp //Return to the processed image
    }
/*
    fun invertBitmap(bitmap: Bitmap): Bitmap {
        sWidth = bitmap.width
        sHeight = bitmap.height
        sPixels = IntArray(sWidth * sHeight)
        bitmap.getPixels(sPixels, 0, sWidth, 0, 0, sWidth, sHeight)
        sIndex = 0
        sRow = 0
        while (sRow < sHeight) {
            sIndex = sRow * sWidth
            sCol = 0
            while (sCol < sWidth) {
                sPixel = sPixels[sIndex]
                sA = sPixel shr 24 and 0xff
                sR = sPixel shr 16 and 0xff
                sG = sPixel shr 8 and 0xff
                sB = sPixel and 0xff
                sR = 255 - sR
                sG = 255 - sG
                sB = 255 - sB
                sPixel =
                    sA and 0xff shl 24 or (sR and 0xff shl 16) or (sG and 0xff shl 8) or (sB and 0xff)
                sPixels[sIndex] = sPixel
                sIndex++
                sCol++
            }
            sRow++
        }
        bitmap.setPixels(sPixels, 0, sWidth, 0, 0, sWidth, sHeight)
        return bitmap
    }
*/
 fun invertBitmap(src: Bitmap): Bitmap {
    val height = src.height
    val width = src.width
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)
    val paint = Paint()
    val matrixGrayscale = ColorMatrix()
    matrixGrayscale.setSaturation(0f)
    val matrixInvert = ColorMatrix()
    matrixInvert.set(
        floatArrayOf(
            -1.0f, 0.0f, 0.0f, 0.0f, 255.0f,
            0.0f, -1.0f, 0.0f, 0.0f, 255.0f,
            0.0f, 0.0f, -1.0f, 0.0f, 255.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        )
    )
    matrixInvert.preConcat(matrixGrayscale)
    val filter = ColorMatrixColorFilter(matrixInvert)
    paint.colorFilter = filter

    canvas.drawBitmap(src, 0f, 0f, paint)


    return bitmap
}

    fun read(bitmap: Bitmap, cropRect: Rect = Rect(), rotation: Int = 0): Result? {
        return read(bitmap, options, cropRect, rotation)
    }

    fun read(bitmap: Bitmap, options: Options, cropRect: Rect = Rect(), rotation: Int = 0): Result? {
        var result = Result()
//        var inverted = invertBitmap(bitmap)
        val status = with(options) {
            read(bitmap, cropRect.left, cropRect.top, cropRect.width(), cropRect.height(), rotation,
                    formats.joinToString(), tryHarder, tryRotate, result)
        }
        return try {
            result.copy(format = Format.valueOf(status!!))
        } catch (e: Throwable) {
            if (status == "NotFound") null else throw RuntimeException(status!!)
        }
    }

    // setting the format enum from inside the JNI code is a hassle -> use returned String instead
    private external fun read(
        bitmap: Bitmap, left: Int, top: Int, width: Int, height: Int, rotation: Int,
        formats: String, tryHarder: Boolean, tryRotate: Boolean,
        result: Result,
    ): String?

    init {
        System.loadLibrary("zxing_android")
    }



    /**
     * Convert the Bitmap to ALPHA_8.  The old one will be recycled.
     */
    fun toAlpha8(src: Bitmap): Bitmap {
        var src = src
        if (!src.hasAlpha()) src = redToAlpha(src)
        val dest = src.copy(Bitmap.Config.ALPHA_8, false)
        src.recycle()
        return dest
    }

    /**
     * Copy the red channel to a new Bitmap's alpha channel.  The old
     * one will be recycled.
     */
    fun redToAlpha(src: Bitmap): Bitmap {
        val dest = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        val matrix = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f)
        val paint = Paint()
        paint.colorFilter = ColorMatrixColorFilter(ColorMatrix(matrix))
        val canvas = Canvas(dest)
        canvas.density = Bitmap.DENSITY_NONE
        canvas.drawBitmap(src, 0f, 0f, paint)
        src.recycle()
        return dest
    }

    /**
     * Create an immutable copy of the given #Bitmap.  Unlike raw
     * Bitmap.copy(), this is safe against a NullPointerException that
     * can occur on old Android versions (1.6) when
     * Bitmap.getConfig()==null.
     */
    fun copy(src: Bitmap, alpha: Boolean): Bitmap {
        var config = src.config
        if (alpha && config != Bitmap.Config.ALPHA_8) return toAlpha8(src)
        if (config == null) /* convert to a format compatible with OpenGL */ config =
            if (src.hasAlpha()) Bitmap.Config.ARGB_8888 else Bitmap.Config.RGB_565
        return src.copy(config, false)
    }
}
