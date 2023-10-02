/* global QUnit */
QUnit.config.autostart = false;

sap.ui.getCore().attachInit(function () {
	"use strict";

	sap.ui.require([
		"deloittetraining/sapui5ap/test/unit/AllTests"
	], function () {
		QUnit.start();
	});
});
