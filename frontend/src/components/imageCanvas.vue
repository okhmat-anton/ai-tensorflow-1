<template>
	<v-container>
		<v-row>
			<v-col>
				<canvas id="mainCanvas" :width="canvasHeightWidth" :height="canvasHeightWidth"
						style="background-color:#eee; border: 1px solid #ccc; margin:10px;"
						@mousedown="mousedown"
						@mousemove="mousemove"
						@mouseup="mouseup"
				>Your browser does not support Canvas
				</canvas>
			</v-col>
			<v-col>
				<v-row>
					<v-btn
						block
						color="primary"
						elevation="5"
						large
						x-large
						class="mt-5"
						@click="recognize()"
					>Predict
					</v-btn>
				</v-row>
				<v-row v-if="getLastResult.image">
					<v-col>
						<img :src="getLastResult.imageForSave" class="resultImg">
						<img :src="getLastResult.imageSmall" class="resultImgSmall">
						<img :src="myImgLikeMnist" class="resultImgSmall" v-if="myImgLikeMnist" :class="'ml-2'">
					</v-col>
					<v-col>
						<h2>Results:</h2>
						<h4 v-for="item in getLastResult.results">{{item.result}}: {{item.prediction}}%</h4>
					</v-col>
				</v-row>
				<v-row v-if="getLastResult.image">
					<v-col
						cols="12"
						sm="5"
						md="5"
						lg="5">
						<v-text-field
							v-model="userSays"
							label="What do you draw?:"
							required
							outlined
						></v-text-field>
					</v-col>
					<v-col
						cols="12"
						sm="3"
						md="3"
						lg="3">
						<v-btn
							@click="saveData"
						>Submit
						</v-btn>
					</v-col>
					<v-col
						cols="12"
						sm="3"
						md="3"
						lg="3">
						<v-btn
							@click="clearCanvas"
						>Clear
						</v-btn>
					</v-col>
				</v-row>
			</v-col>
		</v-row>
	</v-container>
</template>

<script>
import {mapGetters} from "vuex";

export default {
	name: "imageCanvas",
	// mixins:['tfLib'],
	props: ['model'],
	data: () => ({
		userSays: null,
		canvasHeightWidth: 392,
		mouse: {x: 0, y: 0},
		draw: false,
		canvas: null,
		resizedCanvas: null,
		resizedContext: null,
		saveCanvas: null,
		saveContext: null,
		context: null,
		width: null,
		height: null,
		appBar: null,
		isDraw: false,
		image: null,
		imageSmall: null,
		imageForSave: null,
		result: [],
		myImgLikeMnist: null
	}),
	mounted(){
		this.canvas=document.getElementById("mainCanvas");
		this.context=this.canvas.getContext("2d");
		
		this.resizedCanvas=document.createElement("canvas");
		this.resizedCanvas.height=28;
		this.resizedCanvas.width=28;
		this.resizedContext=this.resizedCanvas.getContext("2d");
		
		this.saveCanvas=document.createElement("canvas");
		this.saveCanvas.height=84;
		this.saveCanvas.width=84;
		this.saveContext=this.saveCanvas.getContext("2d");
		
		this.context.lineWidth=this.canvasHeightWidth/20;
		this.width=this.canvas.width;
		this.height=this.canvas.height;
		this.appBar=document.getElementById("appBar")?.getBoundingClientRect();
	},
	computed: {
		...mapGetters([
			'getLastResult',
		])
	},
	methods: {
		mousedown(e){
			this.mouse.x=e.pageX - this.canvas.offsetLeft;
			this.mouse.y=e.pageY - this.canvas.offsetTop - this.appBar.height;
			this.draw=true;
			this.context.beginPath();
			this.context.moveTo(this.mouse.x, this.mouse.y);
			this.isDraw=true;
		},
		mousemove(e){
			if(this.draw === true){
				this.mouse.x=e.pageX - this.canvas.offsetLeft;
				this.mouse.y=e.pageY - this.canvas.offsetTop - this.appBar.height;
				this.context.lineTo(this.mouse.x, this.mouse.y);
				this.context.stroke();
			}
		},
		mouseup(e){
			this.mouse.x=e.pageX - this.canvas.offsetLeft;
			this.mouse.y=e.pageY - this.canvas.offsetTop - this.appBar.height;
			this.context.lineTo(this.mouse.x, this.mouse.y);
			this.context.stroke();
			this.context.closePath();
			this.draw=false;
		},
		/**
		 * подготовка изображения перед тем как сохранить
		 * @param image
		 */
		rebuildImage(image){
			return image;
			// 1. правильно обрезать то что есть https://html.spec.whatwg.org/multipage/canvas.html#dom-context-2d-drawimage
			// 2. привести толщину линий к норме ? наверное такое должен делать AI
		},
		async recognize(){
			if(this.isDraw){
				
				// img.width и img.height будет содержать оригинальные размеры
				// this.canvas.drawImage(img, 0, 0, 28, 28);
				this.resizedContext.clearRect(0, 0, 28, 28);
				this.saveContext.clearRect(0, 0, 84, 84);
				
				this.image=this.canvas.toDataURL();
				this.image=this.rebuildImage(this.image);
				this.saveContext.drawImage(this.canvas, 0, 0, 84, 84);
				this.resizedContext.drawImage(this.canvas, 0, 0, 28, 28);
				this.imageSmall=this.resizedCanvas.toDataURL();
				this.imageForSave=this.saveCanvas.toDataURL();
				let results=await this.predict();
				
				this.result={
					image: this.image,
					imageSmall: this.imageSmall,
					imageForSave: this.imageForSave,
					prediction: results[0].result,
					results: results,
					userSays: null,
					realResult: null,
					date: Date.now()
				};
				
				this.updateResult(this.result);
			}
		},
		saveData(){
			if(this.isDraw){
				if(this.userSays){
					try{
						this.result.userSays=Math.max(0, Math.min(9, this.userSays[0]));
					}catch(e){
					
					}
					this.saveResult(this.result);
				}
				this.clearCanvas();
			}
		},
		/**
		 *  Predict image data
		 * @param result
		 */
		async predict(){
			// 1 подготовить изображение - перевести в строку пикселей отклонения
			let imageDataPixels=this.resizedContext.getImageData(0, 0,
				28,
				28
			);
			let imageDataArray=[];
			for(let j=0; j < imageDataPixels.data.length/4; j++){
				// one channel + invert to MNIST matrix
				let i=(j + 1)*4 - 1;
				imageDataArray[j]=Math.abs(imageDataPixels.data[i])/255;
			}
			
			let image=[imageDataArray];
			
			// let xs=this.$tf.tensor4d(
			// 	this.mnistData.testImages,
			// 	[this.mnistData.testImages.length/this.tfConfig.imageSize(), this.tfConfig.imageHeight, this.tfConfig.imageWidth, 1]);
			
			let xs=this.$tf.tensor4d(
				imageDataArray,
				[imageDataArray.length/(28*28), 28, 28, 1]);
			const output=this.model.predict(xs);
			const predictions=Array.from(output.argMax(1).dataSync());
			
			const canvasTmp = document.createElement('canvas');
			canvasTmp.height=28; canvasTmp.width=28;
			const canvasTmpContext=canvasTmp.getContext('2d');
			const imageData=new ImageData(28, 28);
			
			for(let i=0; i < 28*28; ++i){
				const j=i*4;
				imageData.data[j + 0]=imageDataArray[i]*255;
				imageData.data[j + 1]=imageDataArray[i]*255;
				imageData.data[j + 2]=imageDataArray[i]*255;
				imageData.data[j + 3]=255;
			}
			canvasTmpContext.putImageData(imageData, 0, 0);
			this.myImgLikeMnist = canvasTmp.toDataURL();
			
			return [
				{
					result: predictions[0],
					prediction: '-'
				},
				{
					result: '-',
					prediction: '-'
				},
				{
					result: '-',
					prediction: '-'
				}
			]
		},
		/**
		 * Push to Vuex + save to firebase
		 * @param result
		 */
		saveResult(result){
			let tmp=Object.assign({}, result);
			delete tmp.image;
			this.$db.ref('predictionsResults').push(tmp);
		},
		updateResult(result){
			this.$store.commit('updateLastResult', result);
		},
		clearCanvas(){
			this.updateResult({});
			this.isDraw=false;
			this.userSays=null;
			this.image=null;
			this.context.clearRect(0, 0, this.canvasHeightWidth, this.canvasHeightWidth);
			this.resizedContext.clearRect(0, 0, 28, 28);
			this.saveContext.clearRect(0, 0, 84, 84);
		}
	}
}
</script>

<style scoped>
.resultImg {
	width: 100%;
	border: 1px solid #000;
}

.resultImgSmall {
	width: 28px;
	height: 28px;
	border: 1px solid #000;
}
</style>