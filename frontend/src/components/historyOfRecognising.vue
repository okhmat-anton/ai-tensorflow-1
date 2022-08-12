<template>
	<v-container fluid>
		<v-data-iterator
			v-if="getHistory"
			:items="getHistoryWithSearch()"
			:sort-by="sortBy"
			:sort-desc="sortDesc"
			:items-per-page="16"
			:loading="loading"
			:loading-text="loadingText"
		>
			<template v-slot:header>
				<v-toolbar
					dark
					color="blue darken-3"
					class="mb-1"
				>
					<template>
						<h3>History dataset</h3>
						<v-spacer></v-spacer>
						<v-text-field
							v-model="search"
							clearable
							solo-inverted
							hide-details
							prepend-inner-icon="mdi-magnify"
							label="Search numbers real & prediction"
						></v-text-field>
						<v-spacer></v-spacer>
						<v-btn-toggle
							v-model="sortDesc"
							mandatory
						>
							<v-btn
								large
								depressed
								color="blue"
								:value="false"
							>
								<v-icon>mdi-arrow-up</v-icon>
							</v-btn>
							<v-btn
								large
								depressed
								color="blue"
								:value="true"
							>
								<v-icon>mdi-arrow-down</v-icon>
							</v-btn>
						</v-btn-toggle>
					</template>
				</v-toolbar>
			</template>
			<template v-slot:default="props">
				<v-row>
					<v-col
						v-for="(item, index) in props.items"
						:key="index"
						cols="12"
						sm="3"
						md="3"
						lg="3"
					>
						<history-block :item="item"/>
					</v-col>
				</v-row>
			</template>
			
			<template v-slot:no-data>There is no data in history
			</template>
		</v-data-iterator>
	</v-container>
</template>

<script>
import HistoryBlock from "./historyBlock";
import {mapGetters} from "vuex";

export default {
	name: "historyOfRecognising",
	components: {HistoryBlock},
	data: () => ({
		sortBy: 'date',
		search: '',
		sortDesc: true,
		loading: false,
		_loadingText: 'Loading... Please wait',
		loadingText: '',
	}),
	created(){
		// init firebase history
		this.loading=true;
		this.loadingText=this._loadingText;
		this.$db.ref('predictionsResults').on('value', (data) => {
			if(data.val()){
				this.$store.commit('updateHistory', Object.values(data.val()));
				this.loadingText='';
				this.loading=false;
			}
		});
	},
	computed:{
		...mapGetters([
			'getHistory',
		]),
	},
	methods: {
		getHistoryWithSearch(){
			let items = this.getHistory;
			if(this.search){
				return items.filter(item => {
					let search=parseInt(this.search[0]);
					return (
						item.prediction === search
						|| (item?.userSays === search && !item?.realResult)
						|| item?.realResult === search
					);
				});
			}else{
				return items;
			}
		}
	},
}
</script>

<style scoped>

</style>