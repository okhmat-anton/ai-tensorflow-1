<template>
	<v-container>
		<div id="upPart">
			<image-canvas :model="model" />
		</div>
		<div id="downPart">
			<history-of-recognising />
		</div>
	</v-container>
</template>

<script>
import ImageCanvas from "./imageCanvas";
import HistoryOfRecognising from "./historyOfRecognising";
export default {
	name: "mainPage",
	props: ['model'],
	components: {HistoryOfRecognising, ImageCanvas},
	methods:{
	}
}
</script>

<style scoped>

</style>