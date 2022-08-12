<template>
	<v-container>
		<v-row>
			<v-col>
				<img :src="getImage" class="resultImg" alt="no image"/>
			</v-col>
			<v-col>
				<h4>Results: {{item.results[0].result}}</h4>
				<h6 v-if="userSays">User Says: {{userSays}}</h6>
			</v-col>
		</v-row>
	</v-container>
</template>

<script>
export default {
	name: "historyBlock",
	props: ['item'],
	data: () => ({}),
	computed:{
		
		/*
		 <h6>{{item.results[0].result}}: {{item.results[0].prediction}}%</h6>
				<h6>{{item.results[1].result}}: {{item.results[1].prediction}}%</h6>
				<h6>{{item.results[2].result}}: {{item.results[2].prediction}}%</h6>
		 */
		userSays(){
			return this.item?.userSays;
		},
		getImage(){
			if(this.item.image==undefined){
				return this.item.imageForSave;
			}
			return this.item.image;
		}
	}
}
</script>

<style scoped>
.resultImg {
	width: 100%;
	border: 1px solid #000;
}
</style>