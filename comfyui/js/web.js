import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";


app.registerExtension({
    name: "conceptguidance.web",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.input?.required?.config?.[1]?.concept_guidance_model_config) {
            nodeData.input.required.load_config_yaml = 
                ["LOAD_CONFIG_YAML", nodeData.input.required.config[1].model_name];
        }
    },

    getCustomWidgets(app) {
        return {
            LOAD_CONFIG_YAML(node, inputName, inputData, app) {
                const cb = node.callback;
                const modelName = inputData[1];
                if (!modelName)
                    throw new Error(`Model name not known for node ${node}`);
                const configWidget = node.widgets.find(w => w.name === "config");
                configWidget.callback = () => {
                    loadConfigYaml();
                    if (cb)
                        return cb.apply(this, arguments);
                };

                //const loadConfigBtn = node.addWidget("button", null, null, () => {
                //    loadConfigYaml();
                //});
                //loadConfigBtn.label = "load config";
                //loadConfigBtn.serialize = false;

                async function loadConfigYaml() {
                    const body = new FormData();
                    const confFilename = configWidget.value;
                    body.append("model_name", modelName);
                    body.append("config_filename", confFilename);
                    const resp = await api.fetchApi("/concept-guidance/load-config", {
                        method: "POST",
                        body,
                    });

                    if (resp.status === 200) {
                        const data = await resp.json();
                        const conf = data.config;
                        console.log(`Configuration '${confFilename}' successfully loaded:\n${conf}`);

                        // Fill widget values
                        Object.entries(conf ?? {})
                            .map(([pName, pVal]) => ({
                                w: node.widgets.find(w => w.name === pName),
                                v: pVal,
                            }))
                            .filter(_ => _.w)
                            .forEach(_ => _.w.value = _.v);
                        Object.entries(conf.params ?? {})
                            .map(([pName, pVal]) => ({
                                w: node.widgets.find(w => w.name === `param_${pName}`),
                                v: pVal,
                            }))
                            .filter(_ => _.w)
                            .forEach(_ => _.w.value = _.v);
                    } else {
                        let msg = `Failed to load config: '${confFilename}'`;
                        const serverMsg = await resp.text();
                        if (serverMsg)
                            msg += ` (${serverMsg})`;
                        alert(msg);
                    }
                };
            },
        }
    },
})

