import Vue from 'vue'

export var mutations={
	updateLastResult(state, data){
		Vue.set(state, 'lastResult' , data);
	},
	updateHistory(state, data){
		Vue.set(state, 'historyItems' , data);
	},
	saveAjaxData(state, data){
		if(data.name && data.data){
			Vue.set(state, data.name, data.data);
		}
	}
}