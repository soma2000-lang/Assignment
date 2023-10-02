/*global QUnit*/

sap.ui.define([
	"deloittetraining/sapui5ap/controller/sapui5app.controller"
], function (Controller) {
	"use strict";

	QUnit.module("sapui5app Controller");

	QUnit.test("I should test the sapui5app controller", function (assert) {
		var oAppController = new Controller();
		oAppController.onInit();
		assert.ok(oAppController);
	});

});
