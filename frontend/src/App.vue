<template>
	<v-app>
		<v-app-bar
			app
			color="primary"
			dark id="appBar"
		>
			<div class="d-flex align-center">
				Demonstration of Google Javascript AI recognizing technology by Okhmat Anton & United Ideas Solutions
				Inc.
			</div>
			<v-spacer></v-spacer>

		</v-app-bar>


    <v-main>
      <main-page :model="tfModel"/>
    </v-main>

		<v-overlay :value="overlay"
				   :z-index="10000"
		>
			<v-progress-circular
				indeterminate
				size="64"
			></v-progress-circular>
		</v-overlay>
	</v-app>
</template>

<script>

import MainPage from "./components/mainPage";
import {mapGetters} from "vuex";

export default {
	name: 'App',
	components: {
		MainPage
	},
	
	data: () => ({
		tfvis: null,
		isShowTfvis: false,
		overlay: false,
		
		dialogSaveModel: false,
		dialogStartTrain: false,
	}),
	async mounted(){
      // load model from Firebase - from python export
    this.overlay = true;
      this.tfModel = await this.loadModelFromFile();
      this.tfModel.summary();
    this.overlay = false;
	},
	computed: {
		...mapGetters([
			'getMnistData',
			'getModelWeights'
		])
	},
	methods: {
	}
};
</script>
