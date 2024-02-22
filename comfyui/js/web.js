import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";


app.registerExtension({
    name: "conceptguidance.web",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.input?.required?.config?.[1]?.concept_guidance_model_config) {
            nodeData.input.required.load_config_yaml = 
                ["LOAD_CONFIG_YAML", nodeData.input.required.config[1].model_name];
        }
        if (nodeData?.input?.required?.nouns_list?.[1]?.extracted_nouns_json) {
            nodeData.input.required.extract_nouns = ["EXTRACT_NOUNS"];
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
                        console.log(`Configuration '${confFilename}' successfully loaded:\n${JSON.stringify(conf)}`);

                        // Fill widget values
                        Object.entries(conf ?? {})
                            .map(([pName, pVal]) => ({
                                w: node.widgets.find(w => w.name === pName),
                                v: pVal,
                            }))
                            .filter(_ => _.w)
                            .forEach(_ => { _.w.value = _.v; });
                        Object.entries(conf.params ?? {})
                            .map(([pName, pVal]) => ({
                                w: node.widgets.find(w => w.name === `param_${pName}`),
                                v: pVal,
                            }))
                            .filter(_ => _.w)
                            .forEach(_ => { _.w.value = _.v; });
                    } else {
                        let msg = `Failed to load config: '${confFilename}'`;
                        const serverMsg = await resp.text();
                        if (serverMsg)
                            msg += ` (${serverMsg})`;
                        alert(msg);
                    }
                };

                const loadConfigWidget = node.addWidget("button", inputName, null, () => {
                    loadConfigYaml();
                });
                loadConfigWidget.label = "load config yaml";
                loadConfigWidget.serialize = false;

                return { widget: loadConfigWidget };
            },

            EXTRACT_NOUNS(node, inputName, inputData, app) {
                //const cb = node.callback;
                const promptWidget = node.widgets.find(w => w.name === "prompt");
                //promptWidget.callback = () => {
                //    extractNouns();
                //    if (cb)
                //        return cb.apply(this, arguments);
                //};

                async function extractNouns() {
                    const body = new FormData();
                    const promptVal = promptWidget.value;
                    body.append("prompt", promptVal);
                    const resp = await api.fetchApi("/concept-guidance/extract-nouns", {
                        method: "POST",
                        body,
                    });

                    if (resp.status === 200) {
                        const data = await resp.json();
                        console.log(`Received nouns: ${data.nouns}`);
                        const nounsListAsStr = JSON.stringify(data.nouns, null, 2)

                        const nounsListWidget = node.widgets.find(w => w.name === "nouns_list");
                        nounsListWidget.value = nounsListAsStr;
                    } else {
                        let msg = `Failed to extract nouns from: '${promptVal}'`;
                        const serverMsg = await resp.text();
                        if (serverMsg)
                            msg += ` (${serverMsg})`;
                        alert(msg);
                    }
                };

                const extractNounsWidget = node.addWidget("button", inputName, null, () => {
                    extractNouns();
                });
                extractNounsWidget.label = "extract nouns";
                extractNounsWidget.serialize = false;

                return { widget: extractNounsWidget };
            },
        }
    },
})

