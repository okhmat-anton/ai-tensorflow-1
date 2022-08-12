// Vuex is to slow so use this class without it

export var tfLib={
	data: () => ({
		tfTrainEpochs: 3,
		batchSize: 320,
		tfStopTrain: false,
		mnistData: null,
		tfLossValues: [[], []],
		tfAccuracyValues: [[], []],
		surfaces: {
			inputDataMnist: null,
			inputDataMy: null,
			trainingCartsCurrentResult: null,
			trainingCartsLoss: null,
			trainingCartsAccuracy: null,
			trainingLog: null,
			predictionImages: null,
		},
		surfacesLosChartCanvas: null,
		surfacesAccuracyChartCanvas: null,
		tfModel: null
	}),
	computed: {
		tfConfig(){
			let h=28, w=28;
			let totalSet=65000, trainSet=55000;
			return {
				imageHeight: h,
				imageWidth: w,
				numClasses: 10,
				imageSize: () => {
					return h*w;
				},
				numDatasetElements: totalSet,
				numTrainElements: trainSet,
				numTestElements: () => {
					return totalSet - trainSet
				},
				mnistImagesPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
				mnistLabelsPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
			}
		}
	},
	methods: {
		/**
		 * function get from https://github.com/tensorflow/tfjs-examples/blob/master/mnist/data.js
		 * @returns {Promise<void>}
		 */
		tfStopTrainNow(){
			this.tfStopTrain=true;
		},
		async tfvisSetupBaseData(){
			this.surfaces.inputDataMy=this.tfvis.surface({name: 'My Image Data', tab: 'Data Set'});
			this.surfaces.inputDataMnist=this.tfvis.surface({name: 'Mnist Image Data', tab: 'Data Set'});
			this.surfaces.trainingLog=this.tfvis.surface({name: 'Training Log', tab: 'Training Log'});
			this.surfaces.trainingCartsCurrentResult=this.tfvis.surface({name: 'Current Result', tab: 'Train charts'});
			this.surfaces.trainingCartsLoss=this.tfvis.surface({name: 'Loss Chart', tab: 'Train charts'});
			this.surfaces.trainingCartsAccuracy=null; //this.tfvis.surface({name: 'Accuracy Chart', tab: 'Train charts'});
			/**
			 *  create plot canvas
			 */
			// let canvas =document.createElement('canvas');
			// this.surfaces.trainingCartsLoss.drawArea.appendChild(canvas)
			// canvas =document.createElement('canvas');
			// this.surfaces.trainingCartsAccuracy.drawArea.appendChild(canvas)
			// canvas.className='prediction-canvas';
			this.surfaces.predictionImages=this.tfvis.surface({name: 'Prediction Result Demo', tab: 'Train charts'});
		},
		async tfLoadMnistTrainingData(){
			this.tfUi().logStatus('Loading MNIST data...');
			// Make a request for the MNIST sprited image.
			const img=new Image();
			const canvas=document.createElement('canvas');
			const ctx=canvas.getContext('2d');
			let datasetImages=null;
			const imgRequest=new Promise((resolve, reject) => {
				img.crossOrigin='';
				img.src=this.tfConfig.mnistImagesPath;
				img.onload=() => {
					img.width=img.naturalWidth;
					img.height=img.naturalHeight;
					const datasetBytesBuffer=
						new ArrayBuffer(this.tfConfig.numDatasetElements*this.tfConfig.imageSize()*4);

					const chunkSize=5000;
					canvas.width=img.width;
					canvas.height=chunkSize;

					for(let i=0; i < this.tfConfig.numDatasetElements/chunkSize; i++){
						const datasetBytesView=new Float32Array(
							datasetBytesBuffer, i*this.tfConfig.imageSize()*chunkSize*4,
							this.tfConfig.imageSize()*chunkSize);
						ctx.drawImage(
							img, 0, i*chunkSize, img.width, chunkSize, 0, 0, img.width,
							chunkSize);


						const imageData=ctx.getImageData(0, 0, canvas.width, canvas.height);

						for(let j=0; j < imageData.data.length/4; j++){
							// All channels hold an equal value since the image is grayscale, so
							// just read the red channel.
							datasetBytesView[j]=imageData.data[j*4]/255;
						}
					}
					datasetImages=new Float32Array(datasetBytesBuffer);
					resolve();
				};
			});

			const labelsRequest=fetch(this.tfConfig.mnistLabelsPath);

			const [imgResponse, labelsResponse]=await Promise.all([imgRequest, labelsRequest]);

			let datasetLabels=new Uint8Array(await labelsResponse.arrayBuffer());

			// Slice the the images and labels into train and test sets.
			this.mnistData={
				trainImages: datasetImages.slice(0, this.tfConfig.imageSize()*this.tfConfig.numTrainElements),
				testImages: datasetImages.slice(this.tfConfig.imageSize()*this.tfConfig.numTrainElements),
				trainLabels: datasetLabels.slice(0, this.tfConfig.numClasses*this.tfConfig.numTrainElements),
				testLabels: datasetLabels.slice(this.tfConfig.numClasses*this.tfConfig.numTrainElements),
			}
		},

		/**
		 * Get all training data as a data tensor and a labels tensor.
		 *
		 * @returns
		 *   xs: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
		 *   labels: The one-hot encoded labels tensor, of shape
		 *     `[numTrainExamples, 10]`.
		 */
		tfGetTrainData(){
			const xs=this.$tf.tensor4d(
				this.mnistData.trainImages,
				[this.mnistData.trainImages.length/this.tfConfig.imageSize(), this.tfConfig.imageHeight, this.tfConfig.imageHeight, 1]);
			const labels=this.$tf.tensor2d(
				this.mnistData.trainLabels, [this.mnistData.trainLabels.length/this.tfConfig.numClasses, this.tfConfig.numClasses]);
			return {xs, labels};
		},

		/**
		 * Get all test data as a data tensor and a labels tensor.
		 *
		 * @param {number} numExamples Optional number of examples to get. If not
		 *     provided,
		 *   all test examples will be returned.
		 * @returns
		 *   xs: The data tensor, of shape `[numTestExamples, 28, 28, 1]`.
		 *   labels: The one-hot encoded labels tensor, of shape
		 *     `[numTestExamples, 10]`.
		 */
		tfGetTestData(numExamples){
			let xs=this.$tf.tensor4d(
				this.mnistData.testImages,
				[this.mnistData.testImages.length/this.tfConfig.imageSize(), this.tfConfig.imageHeight, this.tfConfig.imageWidth, 1]);
			let labels=this.$tf.tensor2d(
				this.mnistData.testLabels, [this.mnistData.testLabels.length/this.tfConfig.numClasses, this.tfConfig.numClasses]);

			if(numExamples != null){
				xs=xs.slice([0, 0, 0, 0], [numExamples, this.tfConfig.imageHeight, this.tfConfig.imageWidth, 1]);
				labels=labels.slice([0, 0], [numExamples, this.tfConfig.numClasses]);
			}
			return {xs, labels};
		},
		/**
		 * https://github.com/tensorflow/tfjs-examples/blob/master/mnist/index.js
		 *
		 * Creates a convolutional neural network (Convnet) for the MNIST data.
		 *
		 * @returns {this.$tf.Model} An instance of tf.Model.
		 */
		tfCreateConvModel(){
			// this.tfUi().logStatus('Creating model...');
			// Create a sequential neural network model. tf.sequential provides an API
			// for creating "stacked" models where the output from one layer is used as
			// the input to the next layer.
			const model=this.$tf.sequential();

			// The first layer of the convolutional neural network plays a dual role:
			// it is both the input layer of the neural network and a layer that performs
			// the first convolution operation on the input. It receives the 28x28 pixels
			// black and white images. This input layer uses 16 filters with a kernel size
			// of 5 pixels each. It uses a simple RELU activation function which pretty
			// much just looks like this: __/
			model.add(this.$tf.layers.conv2d({
				inputShape: [this.tfConfig.imageHeight, this.tfConfig.imageWidth, 1],
				kernelSize: 9,
				filters: 48,
				activation: 'relu'
			}));

			// After the first layer we include a MaxPooling layer. This acts as a sort of
			// downsampling using max values in a region instead of averaging.
			// https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
			model.add(this.$tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

			// Our third layer is another convolution, this time with 32 filters.
			model.add(this.$tf.layers.conv2d({kernelSize: 3, filters: 96, activation: 'relu'}));

			// Max pooling again.
			model.add(this.$tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

			// Add another conv2d layer.
			model.add(this.$tf.layers.conv2d({kernelSize: 3, filters: 48, activation: 'relu'}));

			// Now we flatten the output from the 2D filters into a 1D vector to prepare
			// it for input into our last layer. This is common practice when feeding
			// higher dimensional data to a final classification output layer.
			model.add(this.$tf.layers.flatten({}));

			model.add(this.$tf.layers.dense({units: 128, activation: 'relu'}));

			// Our last layer is a dense layer which has 10 output units, one for each
			// output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
			// represent numbers, but it's the same idea if you had classes that
			// represented other entities like dogs and cats (two output classes: 0, 1).
			// We use the softmax function as the activation for the output layer as it
			// creates a probability distribution over our 10 classes so their output
			// values sum to 1.
			model.add(this.$tf.layers.dense({units: 10, activation: 'softmax'}));

			return model;
		},

		/**
		 * Compile and train the given model.
		 * https://github.com/tensorflow/tfjs-examples/blob/master/mnist/index.js
		 *
		 * @param {this.$tf.Model} model The model to train.
		 * @param {onIterationCallback} onIteration A callback to execute every 10
		 *     batches & epoch end.
		 */
		async tfTrain(model, onIteration){
			this.tfUi().logStatus('Training model...');

			// Now that we've defined our model, we will define our optimizer. The
			// optimizer will be used to optimize our model's weight values during
			// training so that we can decrease our training loss and increase our
			// classification accuracy.

			// We are using rmsprop as our optimizer.
			// An optimizer is an iterative method for minimizing an loss function.
			// It tries to find the minimum of our loss function with respect to the
			// model's weight parameters.
			// const optimizer='rmsprop';
			const optimizer='adam';
			// We compile our model by specifying an optimizer, a loss function, and a
			// list of metrics that we will use for model evaluation. Here we're using a
			// categorical crossentropy loss, the standard choice for a multi-class
			// classification problem like MNIST digits.
			// The categorical crossentropy loss is differentiable and hence makes
			// model training possible. But it is not amenable to easy interpretation
			// by a human. This is why we include a "metric", namely accuracy, which is
			// simply a measure of how many of the examples are classified correctly.
			// This metric is not differentiable and hence cannot be used as the loss
			// function of the model.
			model.compile({
				optimizer,
				loss: 'categoricalCrossentropy',
				metrics: ['accuracy'],
			});

			// Batch size is another important hyperparameter. It defines the number of
			// examples we group together, or batch, between updates to the model's
			// weights during training. A value that is too low will update weights using
			// too few examples and will not generalize well. Larger batch sizes require
			// more memory resources and aren't guaranteed to perform better.

			// NOW IT  this.batchSize = 320;

			// Leave out the last 15% of the training data for validation, to monitor
			// overfitting during training.
			const validationSplit=0.15;

			// Get number of training epochs from the UI.
			const trainEpochs=Math.max(1, this.tfTrainEpochs);

			// We'll keep a buffer of loss and accuracy values over time.
			let trainBatchCount=0;

			const trainData=this.tfGetTrainData();
			const testData=this.tfGetTestData();
			const totalNumBatches=
				Math.ceil(trainData.xs.shape[0]*(1 - validationSplit)/this.batchSize)*
				trainEpochs;

			// During the long-running fit() call for model training, we include
			// callbacks, so that we can plot the loss and accuracy values in the page
			// as the training progresses.
			let valAcc;
			await model.fit(trainData.xs, trainData.labels, {
				batchSize: this.batchSize,
				validationSplit,
				epochs: trainEpochs,
				callbacks: {
					onBatchEnd: async(batch, logs) => {
						// if(batch%3 === 0){
							trainBatchCount++;
							this.tfUi().logCurrentStatus(
								`Training... (` +
								`${(trainBatchCount/totalNumBatches*100).toFixed(1)}%` +
								` complete). To stop training, refresh or close page.`);
							this.tfUi().plotLoss(trainBatchCount, logs.loss, 'train');
							this.tfUi().plotAccuracy(trainBatchCount, logs.acc, 'train');
						// }
						if(onIteration && batch%25 === 0){
							onIteration('onBatchEnd', batch, logs);
						}
						if(this.tfStopTrain){
							this.tfStopTrain=false;
							model.stopTraining=true;
						}
						await this.$tf.nextFrame();
					},
					onEpochEnd: async(epoch, logs) => {
						valAcc=logs.val_acc;
						this.tfUi().plotLoss(trainBatchCount, logs.val_loss, 'validation', true);
						this.tfUi().plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
						if(onIteration){
							onIteration('onEpochEnd', epoch, logs);
						}
						//сохранить сетку в локалстор ?
						await model.save('localstorage://tfModel-number-prediction-TMP');
						this.tfUi().logStatus('Эпоха ' + epoch + ' сохранена в память');
						await this.$tf.nextFrame();
					}
				}
			});

			const testResult=model.evaluate(testData.xs, testData.labels);
			const testAccPercent=testResult[1].dataSync()[0]*100;
			const finalValAccPercent=valAcc*100;

			let message=`Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
				`Final test accuracy: ${testAccPercent.toFixed(1)}%`;
			this.tfUi().logCurrentStatus(message);
			this.tfUi().logStatus(message);
		},

		tfSetWeight(model, weights){
			return model.setWeight(weights);
		},
		tfGetWeight(model){
			return model.getWeight();
		},
		/**
		 * ui class from https://github.com/tensorflow/tfjs-examples/blob/master/mnist/ui.js
		 */
		tfUi(){
			return {
				logStatus: (message) => {
					let newLogElement=document.createElement('div');
					newLogElement.innerHTML=message;
					this.surfaces.trainingLog.drawArea.appendChild(newLogElement);
				},
				logCurrentStatus: (message, add=false) => {
					let newLogElement=document.createElement('div');
					newLogElement.innerHTML=message;
					if(!add){
						this.surfaces.trainingCartsCurrentResult.drawArea.innerHTML='';
					}
					this.surfaces.trainingCartsCurrentResult.drawArea.appendChild(newLogElement);
				},
				clearLog: () => {
					this.surfaces.trainingLog.drawArea.innerHTML='';
				},
				trainingLog: (message) => {
					this.tfUi().logStatus(message);
				},
				showTestResults: (batch, predictions, labels) => {
					const testExamples=batch.xs.shape[0];
					let images=this.surfaces.predictionImages.drawArea;
					images.innerHTML='';
					for(let i=0; i < testExamples; i++){
						const image=batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

						const div=document.createElement('div');
						div.className='pred-container';

						const canvas=document.createElement('canvas');
						canvas.className='prediction-canvas';
						this.tfUi().draw(image.flatten(), canvas);

						const pred=document.createElement('div');

						const prediction=predictions[i];
						const label=labels[i];
						const correct=prediction === label;

						pred.className=`pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
						pred.innerText=`pred: ${prediction}`;

						div.appendChild(pred);
						div.appendChild(canvas);

						images.appendChild(div);
					}
				},
				plotLoss: (batch, loss, set, notUpdate) => {
					const series = set === 'train' ? 0 : 1;
					this.tfLossValues[series].push({x: batch, y: loss});
					const seriesEmpty = set !== 'train' ? 0 : 1;
					this.tfLossValues[seriesEmpty].push({x: batch, y: 0});

					if(batch%10){
						this.$tfvis.render.linechart(this.surfaces.trainingCartsLoss.drawArea, {
							values: this.tfLossValues,
							series: ['train', 'validation']
						}, {
							xLabel: 'Batch #',
							yLabel: 'Loss',
							width: 500,
							height: 350,
						});
					}
					this.tfUi().logCurrentStatus(`<br>Last loss: ${loss.toFixed(2)}`, !notUpdate);
				},
				plotAccuracy: (batch, accuracy, set) => {
					set;
					// const series=set === 'train' ? 0 : 1;
					// this.tfAccuracyValues[series].push({x: batch, y: accuracy});
					// const seriesEmpty=set !== 'train' ? 0 : 1;
					// this.tfAccuracyValues[seriesEmpty].push({x: batch, y: 0});
					//
					// this.$tfvis.render.linechart(this.surfaces.trainingCartsAccuracy.drawArea, {
					// 	values: this.tfLossValues,
					// 	series: ['train', 'validation']
					// }, {
					// 	xLabel: 'Batch #',
					// 	yLabel: 'Accuracy',
					// 	width: 500,
					// 	height: 350,
					// });
					this.tfUi().logCurrentStatus(`<br>Last accuracy: ${(accuracy*100).toFixed(1)}%`, true);
				},
				draw: (image, canvas) => {
					const [width, height]=[28, 28];
					canvas.width=width;
					canvas.height=height;
					const ctx=canvas.getContext('2d');
					const imageData=new ImageData(width, height);
					const data=image.dataSync();
					for(let i=0; i < height*width; ++i){
						const j=i*4;
						imageData.data[j + 0]=data[i]*255;
						imageData.data[j + 1]=data[i]*255;
						imageData.data[j + 2]=data[i]*255;
						imageData.data[j + 3]=255;
					}
					ctx.putImageData(imageData, 0, 0);
				},
				getModelTypeId: () => {
					return 'ConvNet';
				},
				showPredictions: (model, countTestExamples=45) => {
					const examples=this.tfGetTestData(countTestExamples);

					// Code wrapped in a tf.tidy() function callback will have their tensors freed
					// from GPU memory after execution without having to call dispose().
					// The tf.tidy callback runs synchronously.
					this.$tf.tidy(() => {
						const output=model.predict(examples.xs);
						// tf.argMax() returns the indices of the maximum values in the tensor along
						// a specific axis. Categorical classification tasks like this one often
						// represent classes as one-hot vectors. One-hot vectors are 1D vectors with
						// one element for each output class. All values in the vector are 0
						// except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
						// output from model.predict() will be a probability distribution, so we use
						// argMax to get the index of the vector element that has the highest
						// probability. This is our prediction.
						// (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
						// dataSync() synchronously downloads the tf.tensor values from the GPU so
						// that we can use them in our normal CPU JavaScript code
						// (for a non-blocking version of this function, use data()).
						const axis=1;
						const labels=Array.from(examples.labels.argMax(axis).dataSync());
						const predictions=Array.from(output.argMax(axis).dataSync());

						this.tfUi().showTestResults(examples, predictions, labels);
					});
				}
			}
		},
	}
}