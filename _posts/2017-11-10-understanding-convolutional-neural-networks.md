Ihttps://youtu.be/GnEmD17kYsEn the past few posts, I've taken a dive into how neural networks work. We even built a neural net that could learn to recognise handwriting by breaking it down into a huge array of the pixels in the image, and representing the colour of the pixel as a value from 0-1.

Our last model got 96% accuracy, but it turns out we can do even better with a different type of neural network that is especially good at images; the convolutional neural network.

In this post we'll explore the concept of convolutional neural networks, how they work, what makes them good at dealing with images and build our own using Tensorflow.

## What is convolution?

Convolution simply means combining two things to form a third thing, which is a modified version of one of the first things. Let's look at how it works in detecting edges in an image...

![Using a filter to detect edges](/assets/images/understanding_convolutional_neural_networks/conv_filter.gif)

Let's take a small sample of our larger image. We'll zoom in on a 5x5 grid of the top corner of our image. Our image will be our first input, which we'll need to convolve with some other input, to create our output. That second input will be called our filter. A filter is simply a smaller grid of weights, and we'll slide this over our 5x5 image sample like in the gif above. You can see the weights written in red. At each step, we'll take each number in our 3x3 window of our image and multiply it by the corresponding weight in our filter cell. So in the top left hand corner, our value is 1, and our filter value is 1, therefore, 1x1 will be 1, and we'll add this to the next value where our image value is 1 but our filter value is 0. We'll repeat the process until we have a total for the values within our filter. This ends up giving us a total of 4. We'll write this in our output, and slide our filter one cell over to the right and repeat to get the next value. Once we reach the end of the row, we'll slide one cell down and back to the left and repeat the process. Repeating this for a 5x5 grid will give us a 3x3 output.

The different values in our filter enhance differences in the image. We can use a filter to detect edges for example by having values in the first and third columns of our 3x3 grid that result in a negative total, and values in the middle column that results in a positive total when passed over an edge. This would result in an output something like this, showing a dark to light to dark edge...

```python
0,1,0
0,1,0
0,1,0
```

That's really all there is to it; filters give us a convenient way to find certain features in an image by their light/dark difference represented numerically. But which filters to use? Well this is something that we will let our neural network learn. During training, it will learn the right features for the job.

## Padding

When we slide our filter over our image, we'll only touch on the corner pixels once, but the middle pixels end up in lots of our windows. This is a problem as this is giving more importance to the middle pixels than the outer ones. We want every pixel to have an even influence in our calculations. Also, our output is now 3x3, so we've lost some size. How do we solve these issues?

The answer is padding. We'll add an extra border of pixels around our image. This means that when we slide over our image with our filter, we not only are able to reach our original edges the same amount of times as we reach the middle pixels, but that our output ends up being the same size as our input. When the output is the same as the input size, this is called *same* padding. When we add no padding, we call this *valid* padding.

![Padding a 6x6 image with 0 pixels](/assets/images/understanding_convolutional_neural_networks/padding.png)

Padding is another hyperparameter that we can tune for our network. It doesn't just have to be one pixel we pad with; a 5x5 image will require a padding of 2 to give a 5x5 output.

## Strides

In our example we took a 5x5 grid and slid our filter over one cell at a time. The distance we move our filter is called a stride. In our example we had a stride of one; moving one cell at a time. Setting the stride to 2 would jump our filter two cells across, and when we reached the end of a row, we would jump two rows down.

## Pooling

## Putting it all together

Right now, we're only detecting simple things, like edges in a 5x5 pixel grid. How do we use this to detect a face or a car in an image?
