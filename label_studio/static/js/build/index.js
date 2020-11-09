/*
 * ATTENTION: The "eval" devtool has been used (maybe by default in mode: "development").
 * This devtool is not neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
/******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ "../dm2/build/static/js/main.js":
/*!**************************************!*\
  !*** ../dm2/build/static/js/main.js ***!
  \**************************************/
/*! default exports */
/*! export DataManager [provided] [no usage info] [missing usage info prevents renaming] */
/*! other exports [not provided] [no usage info] */
/*! runtime requirements: __webpack_exports__ */
/***/ ((__unused_webpack_module, exports) => {


/***/ }),

/***/ "./label_studio/static/js/modules/index.js":
/*!*************************************************!*\
  !*** ./label_studio/static/js/modules/index.js ***!
  \*************************************************/
/*! namespace exports */
/*! exports [not provided] [no usage info] */
/*! runtime requirements: __webpack_require__, __webpack_require__.r, __webpack_exports__, __webpack_require__.* */
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var htx_data_manager__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! htx-data-manager */ \"../dm2/build/static/js/main.js\");\n\n\nconst dmRoot = document.querySelector(\".datamanager\");\n\nconsole.log(dmRoot);\n\nif (dmRoot) {\n  const dm = new htx_data_manager__WEBPACK_IMPORTED_MODULE_0__.DataManager({\n    root: dmRoot,\n    api: {\n      gateway: \"/api\",\n      endpoints: {\n        project: \"/project\",\n        columns: \"/project/columns\",\n        tabs: \"/project/tabs\",\n        updateTab: {\n          path: \"/project/tabs/:tabID\",\n          method: \"post\",\n          headers: {\n            'Content-Type': \"application/json\",\n          },\n        },\n        deleteTab: {\n          path: \"/project/tabs/:tabID\",\n          method: \"delete\",\n        },\n\n        tasks: \"/project/tabs/:tabID/tasks\",\n        annotations: \"/project/tabs/:tabID/annotations\",\n\n        task: \"/tasks/:taskID\",\n        skipTask: \"/tasks/:taskID/completions?was_cancelled=1\",\n        nextTask: \"/project/next\",\n\n        completion: \"/tasks/:taskID/completions/:id\",\n        submitCompletion: {\n          path: \"/tasks/:taskID/completions\",\n          method: \"post\",\n          headers: {\n            'Content-Type': \"application/json\",\n          },\n        },\n        updateCompletion: {\n          path: \"/tasks/:taskID/completions/:completionID\",\n          method: \"post\",\n          headers: {\n            'Content-Type': \"application/json\",\n          },\n        },\n        deleteCompletion: {\n          path: \"/tasks/:taskID/completions/:completionID\",\n          method: \"delete\",\n        },\n      },\n    },\n  });\n\n  console.log(dm);\n}\n\n\n//# sourceURL=webpack://label-studio/./label_studio/static/js/modules/index.js?");

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		if(__webpack_module_cache__[moduleId]) {
/******/ 			return __webpack_module_cache__[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
/******/ 	// startup
/******/ 	// Load entry module
/******/ 	__webpack_require__("./label_studio/static/js/modules/index.js");
/******/ 	// This entry module used 'exports' so it can't be inlined
/******/ })()
;