// Just the integration test driver Job (applied after the infra is ready).
// External vars: ns, image_tag, registry
local fessel = import '../main.libsonnet';

local cfg = fessel.config + {
  namespace: std.extVar('ns'),
  images+: { registry: std.extVar('registry'), tag: std.extVar('image_tag') },
};

fessel.testJob(cfg)
