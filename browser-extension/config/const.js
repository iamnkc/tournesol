/**
 * A configuration file containing constants that can be used by all content
 * scripts.
 *
 * To make this configuration available to a set of content scripts, it must
 * be the first JavaScript file listed in a `content_scripts` element of the
 * `manifest.json`.
 */

// unique HTML id of the extension modal
const EXT_MODAL_ID = 'x-tournesol-modal';
// the value of the CSS property display used to make the modal visible
const EXT_MODAL_VISIBLE_STATE = 'flex';
// the value of the CSS property display used to make the modal invisible
const EXT_MODAL_INVISIBLE_STATE = 'none';

// unique HTML id of the Tournesol login iframe
const IFRAME_TOURNESOL_LOGIN_ID = 'x-tournesol-iframe-login';
