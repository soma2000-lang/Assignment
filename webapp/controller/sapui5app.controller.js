sap.ui.define([
    "sap/ui/core/mvc/Controller"
],
    /**
     * @param {typeof sap.ui.core.mvc.Controller} Controller
     */
    function (Controller) {
        "use strict";

        return Controller.extend("deloitte.training.sapui5ap.controller.sapui5app", {
            onInit: function () {

            },
            onPress: function(event)
            {
                //this.getView().destroy();
                var orouter = sap.ui.core.UIcomponent.getRouterFor(this);
                orouter.navTo("secondpage")
               
                
            }
        });
    
        
    }); 
